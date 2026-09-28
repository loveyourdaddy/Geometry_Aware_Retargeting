import sys
sys.path.append('../0_geometry_aware_retargeting') # 0_geometry_aware_retargeting

from pymovis.vis.core import MeshGL, VertexGL, VAO
from pymovis.vis.model import Model
from pymovis.vis.material import Material
from pymovis.motion.core import Pose
from pymovis.motion.ops.torchmotion import R_fk, R_to_R6, R6_to_R
import glm
import numpy as np
import torch
import os
import pickle
import copy
from pymovis.vis.const import meshSkel_to_sourceSkel
from Geometry.bvh_tree import BVH_Tree
from Geometry.voxel import *
from option_parser import *
from etc.etc import *
from pymovis.motion.data import bvh


class Geometry:
    def __init__(self, args, character=None, Tpose=None, scale=1.0, name=None):
        self.args = args
        self.device = args.device
        self.geo_preprocess = self.args.geo_preprocess 
        if name is None:
            self.name = character.meshes[0].mesh_gl.name
        else:
            self.name = name
        if character is not None:
            self.character = character
            self.meshes = character.meshes
            self.Tpose = Tpose
            
        # two kinds of skeleton: common(22) = mesh skeleton(52), mesh_gl(65)
        # Common skeleton
        self.source_skeleton = self.meshes[0].source_skeleton
        self.num_joints = self.meshes[0].source_skeleton.num_joints
        self.ee_joints = self.meshes[0].source_skeleton.effector_idx
        self.plane_joints = list(set(list(range(0, self.num_joints))) - set(self.ee_joints))

        # T-pose
        self.Tpose = Pose(self.source_skeleton,
                          np.stack([np.eye(3, dtype=np.float32) for _ in range(self.source_skeleton.num_joints)], axis=0),
                          self.source_skeleton.joints[0].offset)

        # Hyper-parameter
        if self.geo_preprocess:
            self.preprocess()
        self.load_data()
        
        self.scale_mesh(scale)

        # bind trf inv
        # self.bind_trf_inv = self._get_bind_trf_inv()
    
    def _get_cid_vid(self):
        c_len, v_len = 0, 0
        cid_to_vids, vid_to_cid = [], []
        for mesh in self.meshes:
            mesh_gl = mesh.mesh_gl
            for _, value in sorted(mesh_gl.control_point_idx_to_vertex_idx.items()):
                vids = [v + v_len for v in value]
                cid_to_vids.append(vids)
            for _, value in sorted(mesh_gl.vertex_idx_to_control_point_idx.items()):
                vid_to_cid.append(value + c_len)
            c_len += len(mesh_gl.control_point_idx_to_vertex_idx)
            v_len += len(mesh_gl.vertex_idx_to_control_point_idx)
        
        vid_to_cid = np.array(vid_to_cid)

        # cid_to_first_vid
        cid_to_first_vid = []
        for i in range(len(cid_to_vids)):
            cid_to_first_vid.append(cid_to_vids[i][0])
        cid_to_first_vid = np.array(cid_to_first_vid)
        # cid_to_first_vid = torch.tensor(cid_to_first_vid, device=cids.device)

        return cid_to_vids, vid_to_cid, cid_to_first_vid

    def _get_vertex_pos_and_normal(self):
        v_position, v_normal = [], []
        for mesh in self.meshes:
            for v in mesh.mesh_gl.vertices:
                v_position.append(v.position)
                v_normal.append(v.normal)

        v_position = np.stack(v_position, axis=0)
        v_normal   = np.stack(v_normal, axis=0)

        return v_position, v_normal

    def _get_control_point_pos_and_normal(self, v_position, v_normal, cid_to_vids):
        c_position, c_normal = [], []
        for vids in cid_to_vids:
            p = v_position[vids[0]] # all vids have same position
            n = v_normal[vids].mean(axis=0)
            n = n / (np.linalg.norm(n) + 1e-8)

            c_position.append(p)
            c_normal.append(n)

        c_position = np.stack(c_position, axis=0)
        c_normal   = np.stack(c_normal, axis=0)
        return c_position, c_normal
    
    def _get_skinning(self, cid_to_vids):
        indices, weights = [], []
        for mesh in self.meshes:
            mesh_gl = mesh.mesh_gl
            for v in mesh_gl.vertices:
                vid1 = self.fbx_vec_to_skeleton_vec(v.skinning_indices1, mesh_gl.idx_to_name) # , mesh_gl.name_to_idx
                vid2 = self.fbx_vec_to_skeleton_vec(v.skinning_indices2, mesh_gl.idx_to_name) # , mesh_gl.name_to_idx
                indices.append(np.array((*vid1, *vid2)))
                weights.append(np.array((*v.skinning_weights1, *v.skinning_weights2), dtype=np.float32))
        
        indices = np.stack(indices, axis=0)
        weights = np.stack(weights, axis=0)
        # cids_indices = indices[c2v]
        # cids_weights = weights[c2v]

        return indices, weights
    
    def _get_joint_data(self, skinning_jids, skinning_weights, v_position, vid_to_cid, cid_to_vids):
        # v_position = v_position.to(self.args.device)
        max_skinning_indices = np.argmax(skinning_weights, axis=1)
        vid_to_jid = skinning_jids[np.arange(len(skinning_jids)), max_skinning_indices]
        cid_to_jid = np.array([vid_to_jid[vids[0]] for vids in cid_to_vids]) # all vids have same joint idx

        # vid, cid, vposition per each joint
        jid_to_vids       = [list() for _ in range(self.num_joints)]
        jid_to_cids       = [set()  for _ in range(self.num_joints)]
        jid_to_vpositions = [list() for _ in range(self.num_joints)]
        jid_to_fids       = [list() for _ in range(self.num_joints)]

        for vid, jid in enumerate(vid_to_jid):
            jid_to_vids[jid].append(vid)
            jid_to_cids[jid].add(vid_to_cid[vid])
            jid_to_vpositions[jid].append(v_position[vid])
        
        for i in range(self.num_joints):
            # if len(jid_to_vpositions[i]) == 0:
            #     continue
            jid_to_vpositions[i] = np.stack(jid_to_vpositions[i], axis=0)
            
        return vid_to_jid, cid_to_jid, jid_to_vids, jid_to_cids, jid_to_vpositions

    def _get_face(self, c_position):
        fid_to_cids = []
        count_cpoint = 0
        for mesh in self.meshes:
            for f in mesh.mesh_gl.triangle_index:
                c0_idx = f[0] + count_cpoint
                c1_idx = f[1] + count_cpoint
                c2_idx = f[2] + count_cpoint
                fid_to_cids.append(np.array([c0_idx, c1_idx, c2_idx]))
            count_cpoint += len(mesh.mesh_gl.control_point_idx_to_vertex_idx)
        
        # f_position
        fid_to_cids = np.stack(fid_to_cids, axis=0)
        f_position = np.sum(c_position[fid_to_cids], axis=1) / 3

        return fid_to_cids, f_position

    def _relate_joint_and_face(self, fid_to_cids, cid_to_jid):
        fid_to_jid = []
        for cids in fid_to_cids:
            jids = cid_to_jid[cids]
            jid = np.argmax(np.bincount(jids))
            fid_to_jid.append(jid)
        fid_to_jid = np.array(fid_to_jid)

        jid_to_fids = [list() for _ in range(self.num_joints)]
        for fid, jid in enumerate(fid_to_jid):
            jid_to_fids[jid].append(fid)
            
        return fid_to_jid, jid_to_fids
    
    def _get_bounding_box(self, position_max_min):
        # 0: +x, +y, +z
        # 1: -x, +y, +z
        # 2: +x, -y, +z
        # 3: -x, -y, +z
        # 4: +x, +y, -z
        # 5: -x, +y, -z
        # 6: +x, -y, -z
        # 7: -x, -y, -z
        boundary_pos = np.concatenate((position_max_min[:, (0, 3, 0, 3, 0, 3, 0, 3)],
                                        position_max_min[:, (1, 1, 4, 4, 1, 1, 4, 4)],
                                        position_max_min[:, (2, 2, 2, 2, 5, 5, 5, 5)]), axis=-1) # (J, 8, 3)
        
        # mid position of face
        face_midpoints = []
        for i in range(self.num_joints):
            aabb = boundary_pos[i]
            mids = np.stack([
                aabb[0] + aabb[2] + aabb[4] + aabb[6], # face 0: +x
                aabb[0] + aabb[1] + aabb[4] + aabb[5], # face 1: +y
                aabb[0] + aabb[1] + aabb[2] + aabb[3], # face 2: +z
                aabb[1] + aabb[3] + aabb[5] + aabb[7], # face 3: -x
                aabb[2] + aabb[3] + aabb[6] + aabb[7], # face 4: -y
                aabb[4] + aabb[5] + aabb[6] + aabb[7], # face 5: -z
            ], axis=0) / 4 # (6, 3)
            face_midpoints.append(mids)

        face_midpoints = np.stack(face_midpoints, axis=0) # (J, 6, 3)


        # quat position of face
        # face_midpoints_ = face_midpoints.reshape(22,6,1,3) # np.expand_dims(face_midpoints, 2)
        # face_quatpoints = np.stack( 
        #                 (face_midpoints_[:, 0] + boundary_pos[:, (0,2,4,6)],
        #                 face_midpoints_[:, 1] + boundary_pos[:, (0,4,5,1)],
        #                 face_midpoints_[:, 2] + boundary_pos[:, (0,1,2,3)],
        #                 face_midpoints_[:, 3] + boundary_pos[:, (1,5,7,3)],
        #                 face_midpoints_[:, 4] + boundary_pos[:, (1,2,3,7)],
        #                 face_midpoints_[:, 5] + boundary_pos[:, (4,6,7,5)]), axis=1) # (J,6,4,3)

        return boundary_pos, face_midpoints # , face_quatpoints

    def _cull_faces(self, face_midpoints): # 
        # Tpose = self.Tpose.poses[0]

        # plane joint
        joint_to_face_midpoints = []
        for i, jid in enumerate(self.plane_joints):
            joint_pos = self.Tpose.global_p[jid]
            parent_idx = self.Tpose.skeleton.parent_idx[jid]
            parent_pos = self.Tpose.global_p[parent_idx] if parent_idx != -1 else np.zeros((3), dtype=np.float32)

            up_axis = np.argmax(np.abs(joint_pos - parent_pos)) # 0:x, 1:y, 2:z
            if up_axis == 0: # +x
                target = face_midpoints[jid, (1, 2, 4, 5)] # 
            elif up_axis == 1: # +y
                target = face_midpoints[jid, (0, 5, 3, 2)]
            else: # +z
                target = face_midpoints[jid, (0, 1, 3, 4)]
            joint_to_face_midpoints.append(target)
        joint_to_face_midpoints = np.stack(joint_to_face_midpoints, axis=0) # (J - ee, 4, 3)

        # ee
        ee_to_face_midpoints = []
        for i, jid in enumerate(self.ee_joints):
            joint_pos  = self.Tpose.global_p[jid]
            parent_idx = self.Tpose.skeleton.parent_idx[jid]
            parent_pos = self.Tpose.global_p[parent_idx] if parent_idx != -1 else np.zeros((3), dtype=np.float32)

            up_axis = np.argmax(np.abs(joint_pos - parent_pos))
            if (joint_pos - parent_pos)[up_axis] < 0:
                up_axis = up_axis + 3 # 0:+x, 1:+y, 2:+z, 3:-x, 4:-y, 5:-z
            
            # right hand rotation for +, left hand rotation for -
            if up_axis == 0: # +x
                target = face_midpoints[jid, (1, 2, 4, 5, 0)]
            elif up_axis == 1:
                target = face_midpoints[jid, (0, 5, 3, 2, 1)]
            elif up_axis == 2:
                target = face_midpoints[jid, (0, 1, 3, 4, 2)]
            elif up_axis == 3:
                target = face_midpoints[jid, (1, 2, 4, 5, 3)]
            elif up_axis == 4:
                target = face_midpoints[jid, (0, 5, 3, 2, 4)]
            else:
                target = face_midpoints[jid, (0, 1, 3, 4, 5)]
            ee_to_face_midpoints.append(target)
        ee_to_face_midpoints = np.stack(ee_to_face_midpoints, axis=0) # (ee, 5, 3)

        return joint_to_face_midpoints, ee_to_face_midpoints

    def _get_anchor_vids(self, jid_to_vids, v_position, joint_to_face_midpoints, ee_to_face_midpoints, boundary_pos): #  
        if self.name == "SMPLx_fat":
            smplx_path = './preprocess/{}/data.pkl'.format("SMPLx")
            with open(smplx_path, 'rb') as f:
                data = pickle.load(f)
            return data['anchor_vids']
        
        anchor_vids = []
        # plane joint
        for i, jid in enumerate(self.plane_joints):
            
            vids = jid_to_vids[jid]
            if len(vids) == 0:
                continue

            v_pos = v_position[vids]
            f_midpos = joint_to_face_midpoints[i]

            for fid in range(4):
                dist = np.linalg.norm(v_pos - f_midpos[fid], axis=1)
                min_vid = np.argmin(dist)
                anchor_vids.append(vids[min_vid])
            
        # end effectors
        for i, jid in enumerate(self.ee_joints):
            vids = jid_to_vids[jid]
            if len(vids) == 0:
                continue

            v_pos = v_position[vids]
            f_midpos = ee_to_face_midpoints[i]

            for fid in range(5):
                dist = np.linalg.norm(v_pos - f_midpos[fid], axis=1)
                min_vid = np.argmin(dist)
                anchor_vids.append(vids[min_vid])

        anchor_vids = np.array(anchor_vids) # 17*4 + 5*5 = 68+25 = 93

        return anchor_vids

    def _get_descriptor_vids(self, jid_to_vids, jid_to_fids, c_position):
        if self.name == "SMPLx_fat":
            smplx_path = './preprocess/{}/data.pkl'.format("SMPLx")
            with open(smplx_path, 'rb') as f:
                data = pickle.load(f)
            return data['descriptor_vids'], data['root_descriptor_vids']
        
        # area per joint
        jid_area= [0] * self.num_joints
        triangle_index = []
        for mesh in self.meshes:
            triangle_index.extend(mesh.mesh_gl.triangle_index)
        for j, fids in enumerate(jid_to_fids):
            for fid in fids:
                tri_cids = triangle_index[fid] 
                tri_cpos = c_position[tri_cids]
                jid_area[j] += np.linalg.norm(np.cross(tri_cpos[1]-tri_cpos[0], tri_cpos[2]-tri_cpos[0]))/2

        # total descriptor points 
        len_cid = len(c_position)
        ratio = 10
        num_descriptor = int(len_cid/ratio)
        
        # number of vids per joint
        sum_area = 0
        for area in jid_area:
            sum_area += area
        num_desc_perjoint = (jid_area/sum_area * num_descriptor).astype(int)
        
        import random
        descriptor_vids = []
        root_descriptor_vids = []
        for j, vids in enumerate(jid_to_vids):
            if len(vids) == 0:
                continue
            vids = random.sample(vids, num_desc_perjoint[j])
            descriptor_vids += vids 
            if j==0:
                root_descriptor_vids += vids
        descriptor_vids.sort()
        return np.array(descriptor_vids), root_descriptor_vids
    
    def _get_bind_trf_inv(self):
        trfs = []
        for i in range(self.num_joints):
            joint_name = self.source_skeleton.joints[i].name
            mesh_gl = self.meshes[0].mesh_gl # assume that all meshes have same skeleton
            jid = mesh_gl.name_to_idx[joint_name]

            bind_trf_inv = mesh_gl.bind_trf_inv[jid]
            bind_trf_inv = torch.from_numpy(np.array(bind_trf_inv)).to(self.device)
            trfs.append(bind_trf_inv)

        return torch.stack(trfs, dim=0) # (J, 4, 4)
    
    
    """ preprocess """
    def load_data(self):
        # saved geometry data 
        path = './preprocess/{}/{}.pkl'.format(self.name, "data")
        # if there is no pkl file, make new one
        if os.path.isfile(path) is False:
            print(f"make new Geometry data.pkl for {self.name}")
            self.preprocess()
        # open pkl
        with open(path, 'rb') as f:
            pkl = pickle.load(f)

        # load from pkl
        self.cid_to_vids = pkl["cid_to_vids"]
        self.vid_to_cid  = torch.from_numpy(pkl["vid_to_cid"]).to(self.device)
        self.cid_to_first_vid = torch.from_numpy(pkl["cid_to_first_vid"]).to(self.device)
        self.fid_to_cids = torch.from_numpy(pkl["fid_to_cids"]).to(self.device)

        self.c_position = torch.from_numpy(pkl["c_position"]).to(self.device)
        self.v_position = torch.from_numpy(pkl["v_position"]).to(self.device)
        self.c_normal   = torch.from_numpy(pkl["c_normal"]).to(self.device)
        self.v_normal   = torch.from_numpy(pkl["v_normal"]).to(self.device)
        self.f_position = torch.from_numpy(pkl["f_position"]).to(self.device)

        self.skinning_indices1 = torch.from_numpy(pkl["skinning_indices1"]).to(self.device)
        self.skinning_indices2 = torch.from_numpy(pkl["skinning_indices2"]).to(self.device)
        self.skinning_weights1 = torch.from_numpy(pkl["skinning_weights1"]).to(self.device)
        self.skinning_weights2 = torch.from_numpy(pkl["skinning_weights2"]).to(self.device)

        self.perjoint_vids      = pkl["perjoint_vids"]
        self.perjoint_cids      = pkl["perjoint_cids"]
        self.perjoint_fids      = pkl["perjoint_fids"]
        self.perjoint_min       = torch.from_numpy(pkl["perjoint_min"]).to(self.device)
        self.perjoint_max       = torch.from_numpy(pkl["perjoint_max"]).to(self.device)
        self.perjoint_vposition = pkl["perjoint_vposition"]
        for i in range(len(self.perjoint_vposition)):
            self.perjoint_vposition[i] = torch.from_numpy(self.perjoint_vposition[i]).to(self.device)

        self.descriptor_vids = pkl["descriptor_vids"]
        self.root_descriptor_vids = pkl["root_descriptor_vids"]
        # self.aabb_anchor_vids = pkl["aabb_anchor_vids"]
        self.boundary_pos = pkl["boundary_pos"]
        
        # geo
        self.name_to_idx = pkl["name_to_idx"]
        self.bind_trf_inv = torch.from_numpy(pkl["bind_trf_inv"])
        self.anchor_vids = torch.from_numpy(pkl["anchor_vids"])
        self.anchor_vpos = torch.from_numpy(pkl["anchor_vpos"])
        self.bone_offset = torch.from_numpy(pkl["bone_offset"])
        self.names = pkl["names"]
        self.parents = torch.from_numpy(pkl["parents"])

        # additional data
        self.c_length = len(self.cid_to_vids)
        self.v_length = len(self.vid_to_cid)
        self.f_length = len(self.fid_to_cids)

        self.cid_max_len, self.vid_max_len = 0, 0
        for cids in self.perjoint_cids:
            self.cid_max_len = max(self.cid_max_len, len(cids))
        for vids in self.perjoint_vids:
            self.vid_max_len = max(self.vid_max_len, len(vids))

        self.height = torch.max(self.v_position[:, 1]).item()
        
        """ bvh for 22 joints """
        self.bvh_tree = BVH_Tree(self.args, self)

    def preprocess(self):
        """ 1. cid <-> vid  """
        cid_to_vids, vid_to_cid, cid_to_first_vid = self._get_cid_vid()

        """ 2. position, normal """
        v_position, v_normal = self._get_vertex_pos_and_normal()
        c_position, c_normal = self._get_control_point_pos_and_normal(v_position, v_normal, cid_to_vids)

        """ 3. skinning data """
        skinning_jids, skinning_weights = self._get_skinning(cid_to_vids)

        """ 4. perjoint vids, cids, vposition """
        vid_to_jid, cid_to_jid, jid_to_vids, jid_to_cids, jid_to_vpositions = self._get_joint_data(skinning_jids, skinning_weights, v_position, vid_to_cid, cid_to_vids)

        """ 5. face """
        fid_to_cids, f_position = self._get_face(c_position)
        fid_to_jid, jid_to_fids = self._relate_joint_and_face(fid_to_cids, cid_to_jid)

        """ 6. anchor vid for each joint"""
        jid_to_vposition_max = np.stack([np.max(j2v, axis=0) for j2v in jid_to_vpositions], axis=0)
        jid_to_vposition_min = np.stack([np.min(j2v, axis=0) for j2v in jid_to_vpositions], axis=0)
        jid_to_max_min = np.concatenate([jid_to_vposition_max, jid_to_vposition_min], axis=1)[:, :, None] # (J, 6, 1)

        boundary_pos, face_midpoints = self._get_bounding_box(jid_to_max_min) # (J, 24)
        joint_to_face_midpoints, ee_to_face_midpoints = self._cull_faces(face_midpoints)
        # anchor 
        anchor_vids = self._get_anchor_vids(jid_to_vids, v_position, joint_to_face_midpoints, ee_to_face_midpoints, boundary_pos)
        anchor_vpos = v_position[anchor_vids]
        
        """ 7. descriptor point """ # random sampled vertex 
        descriptor_vids, root_descriptor_vids = self._get_descriptor_vids(jid_to_vids, jid_to_fids, c_position) # randmom sampled RD (1245)
        
        # geo data
        name_to_idx = self.meshes[0].mesh_gl.name_to_idx
        bind_trf_inv = self.meshes[0].mesh_gl.bind_trf_inv
        np_bind_trf_inv = []
        for trf in bind_trf_inv:
            np_bind_trf_inv.append(np.array(trf).T) 
        np_bind_trf_inv = np.stack(np_bind_trf_inv, axis=0)
        # skeleton info 
        offsets = []
        names = []
        for joint in self.source_skeleton.joints:
            offsets.append(joint.offset)
            names.append(joint.name)
        bone_offset = torch.tensor(np.array(offsets))
        parents = self.source_skeleton.parent_idx
        
        os.makedirs('./preprocess/{}'.format(self.name), exist_ok=True)
        data = {
            "cid_to_vids": cid_to_vids,
            "vid_to_cid": vid_to_cid,
            "cid_to_first_vid": cid_to_first_vid,
            "fid_to_cids": fid_to_cids,

            "c_position": c_position,
            "v_position": v_position,
            "c_normal": c_normal,
            "v_normal": v_normal,
            "f_position": f_position,

            "skinning_indices": skinning_jids,
            "skinning_weights": skinning_weights,
            "skinning_indices1": skinning_jids[:, :4],
            "skinning_weights1": skinning_weights[:, :4],
            "skinning_indices2": skinning_jids[:, 4:],
            "skinning_weights2": skinning_weights[:, 4:],

            "perjoint_vids": jid_to_vids,
            "perjoint_cids": jid_to_cids,
            "perjoint_fids": jid_to_fids,
            "perjoint_vposition": jid_to_vpositions,
            "perjoint_min": jid_to_vposition_min,
            "perjoint_max": jid_to_vposition_max,

            # "aabb_anchor_vids": aabb_anchor_vids, # network 
            "descriptor_vids": descriptor_vids, # random anchor  
            "root_descriptor_vids": root_descriptor_vids, 
            "boundary_pos": boundary_pos,
            
            # character, geo data
            "name_to_idx": name_to_idx,
            "bind_trf_inv": np.array(np_bind_trf_inv),
            "anchor_vids": np.array(anchor_vids), 
            "anchor_vpos": np.array(anchor_vpos), 

            "bone_offset": np.array(bone_offset),
            "names": np.array(names),
            "parents": np.array(parents),
        }

        with open('./preprocess/{}/{}.pkl'.format(self.name, "data"), 'wb') as f:
            pickle.dump(data, f)
            
        print(">Preprocess done")
        
    def scale_mesh(self, scale):
        self.v_position *= scale
        self.c_position *= scale
        
    def renderable_model(self):
        mesh = MeshGL()
        mesh.name = self.name

        v_position        = self.v_position.cpu().numpy()
        v_normal          = self.v_normal.cpu().numpy()

        vertices = []
        for i in range(len(v_position)):
            vertices.append(VertexGL(glm.vec3(v_position[i]),
                                     glm.vec3(v_normal[i]),
                                     material_id=0))
        mesh.vertices = vertices
        mesh.indices = [i for i in range(len(vertices))]
        mesh.is_skinned = False
        mesh.vao = VAO.from_vertex_array(mesh.vertices, mesh.indices)

        material = [Material()]
        return Model([[mesh, material]])
    

    def renderable_model_by_pose(self, pose):
        local_R = torch.from_numpy(pose.local_R).to(self.device)
        root_p  = torch.from_numpy(pose.root_p).to(self.device)
        global_R, global_p = R_fk(local_R, root_p, pose.skeleton)

        vertex_trfs = torch.empty([self.num_joints, 4, 4]).to(self.device)
        bind_trf_inv = self._get_bind_trf_inv()
        for i in range(self.num_joints):
            trf = torch.cat([global_R[i], global_p[i].unsqueeze(1)], dim=1)
            trf = torch.cat([trf, torch.tensor([[0, 0, 0, 1]], dtype=trf.dtype, device=trf.device)], dim=0)
            vertex_trfs[i] = torch.matmul(trf, bind_trf_inv[i])
        
        # transform vertices
        trfs = vertex_trfs[self.skinning_indices]
        trfs = torch.sum(trfs * self.skinning_weights.unsqueeze(-1).unsqueeze(-1), dim=1) # (V, 4, 4)

        v_position = torch.cat([self.v_position, torch.ones([self.v_length, 1]).to(self.device)], dim=1) # (V, 4)
        v_position = torch.matmul(trfs, v_position.unsqueeze(-1)).squeeze(-1) # (V, 4)

        v_normal = torch.cat([self.v_normal, torch.zeros([self.v_length, 1]).to(self.device)], dim=1) # (V, 4)
        v_normal = torch.matmul(trfs.inverse().transpose(-1, -2), v_normal.unsqueeze(-1)).squeeze(-1) # (V, 4)

        mesh = MeshGL()
        mesh.name = self.name

        vertices = []
        for i in range(len(v_position)):
            vertices.append(VertexGL(glm.vec3(v_position[i, :3].cpu().numpy()),
                                     glm.vec3(v_normal[i, :3].cpu().numpy()),
                                     material_id=0))
        mesh.vertices = vertices
        mesh.indices = [i for i in range(len(vertices))]
        mesh.is_skinned = False
        mesh.vao = VAO.from_vertex_array(mesh.vertices, mesh.indices)

        material = [Material()]
        return Model([[mesh, material]])

    def renderable_model(self):
        v_position = self.v_position.cpu().numpy()
        v_normal   = self.v_normal.cpu().numpy()
        skinning_indices1 = self.skinning_indices1
        skinning_indices2 = self.skinning_indices2
        skinning_weights1 = self.skinning_weights1 # .cpu().numpy()
        skinning_weights2 = self.skinning_weights2 

        vertices = []
        for i in range(len(v_position)):
            vertices.append(VertexGL(glm.vec3(v_position[i]),
                                     glm.vec3(v_normal[i]),
                                     material_id=0,
                                     skinning_indices1=glm.ivec4(*skinning_indices1[i].ravel()),
                                     skinning_weights1=glm.vec4 (*skinning_weights1[i].ravel()),
                                     skinning_indices2=glm.ivec4(*skinning_indices2[i].ravel()),
                                     skinning_weights2=glm.vec4 (*skinning_weights2[i].ravel())))
        
        mesh = MeshGL()
        mesh.vertices = vertices
        mesh.indices = [i for i in range(len(vertices))]
        mesh.is_skinned = True
        mesh.joint_order = [self.source_skeleton.joints[i].name for i in range(self.num_joints)]
        mesh.name_to_idx = self.source_skeleton.idx_by_name
        mesh.idx_to_name = self.source_skeleton.name_by_idx
        mesh.bind_trf_inv = [glm.mat4(*trf_inv.T.ravel()) for trf_inv in self._get_bind_trf_inv().cpu().numpy()]
        mesh.vao = VAO.from_vertex_array(mesh.vertices, mesh.indices)

        material = [Material(albedo=glm.vec3(0.5))]

        # model
        model = Model([[mesh, material]], self.source_skeleton)
        model.set_source_skeleton(self.source_skeleton, "")
        return model
        
    def fbx_vec_to_skeleton_vec(self, fbx_vec, idx_to_name): 
        def fbx_skeleton_idx_to_source_skeleton_idx(j):
            fbx_name = idx_to_name[j]
            if 'mixamorig:' in fbx_name:
                fbx_name = fbx_name.replace('mixamorig:','')
            if 'mixamorig6:' in fbx_name:
                fbx_name = fbx_name.replace('mixamorig6:','')
            if 'mixamorig9:' in fbx_name:
                fbx_name = fbx_name.replace('mixamorig9:','')

            # fbx joint -> bind to the nearest skeleton joint
            name_to_idx = self.meshes[0].source_skeleton.idx_by_name
            for key, values in meshSkel_to_sourceSkel.items():
                for _, value in enumerate(values):
                    if fbx_name in value:
                        source_idx = name_to_idx[key]
                        return source_idx
            raise ValueError("There is no joint name in skeleton_dic")

        v_skinning_indices = []
        for idx in fbx_vec:
            if idx == -1:
                v_skinning_indices.append(-1)
                continue
            source_idx = fbx_skeleton_idx_to_source_skeleton_idx(idx)
            v_skinning_indices.append(source_idx)
        
        return v_skinning_indices
    
    """ Deformation """
    def deform_by_pose(self, local_R, root_p):
        """
        Args:
            local_R: (...., J, 3, 3)
            root_p: (...., 3)

        Returns:
            v_position: (...., V, 3)
            v_normal: (...., V, 3)
        """
        dtype, device = local_R.dtype, local_R.device

        # 4x4 transformation matrix
        global_R, global_p = R_fk(local_R, root_p, self.source_skeleton)        # (...., J, 3, 3), (...., J, 3)
        trf = torch.cat([global_R, global_p.unsqueeze(-1)], dim=-1)             # (...., J, 3, 4)
        homogen = torch.tensor([[[[0, 0, 0, 1]]]], dtype=dtype, device=device)  # (1, 1, 1, 4)
        trf = torch.cat([trf, homogen.repeat(trf.shape[:-1] + (1, 1))], dim=-2) # (...., J, 4, 4)
        

    """ Set pose """
    # Set mesh_global_R (motion joints)
    def set_pose_by_source_batch_frame(self, local_R, root_p):
        # target
        num_batch, num_frame, _, _, _ = local_R.shape
        self.mesh_global_R = torch.zeros(num_batch, num_frame, self.num_joints, 4, 4).to(local_R.device) # batch added

        global_R, global_p = R_fk(local_R, root_p, self.source_skeleton)
        self.global_R = global_R
        self.global_p = global_p
        for i in range(self.source_skeleton.num_joints):
            # world trf 
            world_trf = torch.cat([global_R[..., i, :, :], global_p[..., i, :, None]], axis=-1)
            world_trf = torch.cat([world_trf, torch.tensor([[[0, 0, 0, 1]]]).repeat(num_batch, num_frame, 1, 1).to(local_R.device)], axis=-2)

            # (i -> mesh jid) for binding
            joint_name = self.source_skeleton.joints[i].name
            for mesh in self.meshes:
                mesh_gl = mesh.mesh_gl
                if joint_name not in mesh_gl.name_to_idx.keys():
                    joint_name_ = 'mixamorig:'+joint_name 
                    if joint_name_ not in mesh_gl.name_to_idx.keys():
                        print("There is no joint name in mesh_gl: ", joint_name_)
                        continue
                else:
                    joint_name_ = joint_name
                mesh_jid = mesh_gl.name_to_idx[joint_name_]

                # bind_trf_inv: global trf of the joint (unique per joint, so applied only once)
                bind_trf_inv = mesh_gl.bind_trf_inv[mesh_jid]
                bind_trf_inv = torch.cat((torch.tensor([bind_trf_inv[0]]),
                                        torch.tensor([bind_trf_inv[1]]),
                                        torch.tensor([bind_trf_inv[2]]),
                                        torch.tensor([bind_trf_inv[3]])), axis=0)
                # print(f"{mesh_jid}, {bind_trf_inv}")
                
                # if i==0:
                #     print("{} \n{}".format(i, bind_trf_inv))
                bind_trf_inv = bind_trf_inv.repeat(num_frame, 1, 1).to(local_R.device)
                # update 
                self.mesh_global_R[..., i, :, :] += torch.matmul(world_trf, bind_trf_inv.transpose(-2, -1))
                break 
            
            
    """ Joint Matrix """
    # skinning_joint, skinning_weight
    def get_onedim_lbsModel_wo_batch(self, joint, weight, frames): 
        len_frame, len_cid = self.len_frame, self.len_cid

        # Set -1 weight to 0
        nega_idx = torch.where(joint == -1)[0]
        if len(nega_idx) == len_frame * len_cid:
            return torch.zeros([len_frame, len_cid, 4, 4], device=joint.device)
        
        joint_without_negatives = update_index(joint, 0, nega_idx)
        joint = joint_without_negatives.long().reshape(len_frame, len_cid) # Dimension change: f, c

        # Rotation matrix of joint
        global_R = torch.zeros([len_frame, len_cid, 4, 4], device=joint.device) # f,c,4,4
        global_R_ = torch.index_select(self.mesh_global_R, 0, frames)
        for f in range(len_frame):
            global_R[f] = torch.index_select(global_R_[f], 0, joint[f])
        global_R = global_R.reshape(len_frame * len_cid, 4, 4)

        # Skinning weight of cid
        weight_without_negatives = update_index(weight, 0, nega_idx)

        # lbs
        lbsModel = torch.zeros([len_frame * len_cid, 4, 4], device=joint.device) # f*c,4,4
        lbsModel += global_R * weight_without_negatives.unsqueeze(-1).unsqueeze(-1)  # f*c,4,4

        return lbsModel.reshape(len_frame, len_cid, 4, 4)  # f,c,4,4
    
    # skinning_joint, skinning_weight
    def get_onedim_lbsModel(self, joint, weight, batches, frames):
        lbs_shape = (1,) * (len(joint.shape)) + (4, 4)
        
        # Set -1 weight to 0f
        joint = joint.long()
        nega_idx = torch.where(joint == -1)
        if len(nega_idx[0]) == torch.numel(joint): 
            return torch.zeros_like(joint, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).repeat(*lbs_shape).to(joint.device) # lbsModel
        if len(nega_idx[0]) != 0:
            joint = update_index(joint, 0, nega_idx)

        # Rotation matrix of joint
        global_R = self.mesh_global_R[batches, frames, joint]

        # Skinning weight of cid
        if len(nega_idx[0]) != 0:
            weight = update_index(weight, 0, nega_idx)

        # lbs
        lbsModel = global_R * weight.unsqueeze(-1).unsqueeze(-1)

        return lbsModel

    def get_lbsModel(self, jids_of_cids, weights_of_cids, frames): # batches, 
        """
        Update only non negative index 
        Input, output: [fid, cid, 4, 4]
        """
        # Get update index 
        negative_idx = torch.where(jids_of_cids == -1)
        if len(negative_idx) == self.len_col_jids * self.cid_max_len:
            return torch.zeros([self.len_col_jids, self.cid_max_len, 4, 4], device=jids_of_cids.device)

        # Global rotation matrix 
        # exception handling: negative index (-1)
        jids_of_cids = remove_negative_index(jids_of_cids, negative_idx).long()
        
        global_R_of_cids = torch.zeros([self.len_col_jids, self.cid_max_len, 4, 4], device=jids_of_cids.device) # f,c,4,4
        mesh_global_R_by_frames = torch.index_select(self.mesh_global_R, 0, frames)
        for f in range(self.len_col_jids):
            global_R_of_cids[f] = torch.index_select(mesh_global_R_by_frames[f], 0, jids_of_cids[f]) 

        # Skinning weight 
        cids_weights_without_negatives = remove_negative_index(weights_of_cids, negative_idx)

        # lbs
        lbsModel = global_R_of_cids * cids_weights_without_negatives.unsqueeze(-1).unsqueeze(-1)  # f*c,4,4

        return lbsModel

    """ position """
    
    def get_positions_from_vids(self, vids, batches, frames): # input [shape] -> output [shape, 4]  # (shape should be same)
        # vids = vids.long()
        lbs_shape = (1,) * (len(vids.shape)) + (4, 4)
        lbsModel = torch.zeros_like(vids, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).repeat(*lbs_shape).to(vids.device)
        positions_shape = (1,) * (len(vids.shape)) + (3,)
        v_positions = torch.zeros_like(vids, dtype=torch.float).unsqueeze(-1).repeat(*positions_shape).to(vids.device)
        one_shape = (1,) * (len(vids.shape)) + (1,)
        one_tensor = torch.ones_like(vids, dtype=torch.float).unsqueeze(-1).repeat(*one_shape).to(vids.device)
        
        # get weights 
        joint_ids1 = self.skinning_indices1[vids]
        weights1   = self.skinning_weights1[vids]
        joint_ids2 = self.skinning_indices2[vids]
        weights2   = self.skinning_weights2[vids]

        # lbsModel [shape,4,4]
        lbsModel1 = torch.zeros_like(lbsModel, device=lbsModel.device)
        lbsModel2 = torch.zeros_like(lbsModel, device=lbsModel.device)
        for i in range(4):
            lbsModel1 += self.get_onedim_lbsModel(joint_ids1[..., i], weights1[..., i], batches, frames)
            lbsModel2 += self.get_onedim_lbsModel(joint_ids2[..., i], weights2[..., i], batches, frames)
        lbsModel = lbsModel1 + lbsModel2

        # position [shape,3]
        uni_vals = torch.unique(vids)
        for val in uni_vals:
            vid_indices = torch.where(vids == val) #[0]
            v_positions[vid_indices] = self.v_position[int(val)]
        pos = torch.cat((v_positions, one_tensor), dim=-1).unsqueeze(-1)

        # get moved position
        fPosition = torch.matmul(lbsModel, pos).squeeze(-1).to(vids.device)
        
        return fPosition[..., :3]
    
    # def get_positions_from_cids(self, cids, batches, frames): # input [shape] -> output [shape, 4]  # (shape should be same)
    #     # vids = vids.long()
    #     lbs_shape = (1,) * (len(cids.shape)) + (4, 4)
    #     lbsModel = torch.zeros_like(cids, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).repeat(*lbs_shape).to(cids.device)
    #     positions_shape = (1,) * (len(cids.shape)) + (3,)
    #     v_positions = torch.zeros_like(cids, dtype=torch.float).unsqueeze(-1).repeat(*positions_shape).to(cids.device)
    #     one_shape = (1,) * (len(cids.shape)) + (1,)
    #     one_tensor = torch.ones_like(cids, dtype=torch.float).unsqueeze(-1).repeat(*one_shape).to(cids.device)
        
    #     # get weights 
    #     joint_ids1 = self.skinning_indices1[cids]
    #     weights1   = self.skinning_weights1[cids]
    #     joint_ids2 = self.skinning_indices2[cids]
    #     weights2   = self.skinning_weights2[cids]

    #     # lbsModel [shape,4,4]
    #     lbsModel1 = torch.zeros_like(lbsModel, device=lbsModel.device)
    #     lbsModel2 = torch.zeros_like(lbsModel, device=lbsModel.device)
    #     for i in range(4):
    #         lbsModel1 += self.get_onedim_lbsModel(joint_ids1[..., i], weights1[..., i], batches, frames)
    #         lbsModel2 += self.get_onedim_lbsModel(joint_ids2[..., i], weights2[..., i], batches, frames)
    #     lbsModel = lbsModel1 + lbsModel2

    #     # position [shape,3]
    #     v_positions = self.c_position[cids]
    #     pos = torch.cat((v_positions, one_tensor), dim=-1).unsqueeze(-1)

    #     # get moved position
    #     fPosition = torch.matmul(lbsModel, pos).squeeze(-1).to(cids.device)
        
    #     return fPosition[..., :3]
    
    def cids_to_vids(self, cids):
        vids = torch.zeros_like(cids)
        for i in range(cids.shape[0]): # Too slow 
            for j in range(cids.shape[1]):
                vids[i][j] = self.cid_to_vids[cids[i][j]][0]

        return vids
    
    # some frame, some cids TODO: remove, and use vids 
    def get_positions_from_colliding_cids(self, cids, batches, frames): # [col_num, cids] [num fid col, cids in fid]
        len_col, len_cid = cids.shape
        
        # exception handling for -1
        negative_index = torch.where(cids == -1)
        cids = update_index(cids, 0, negative_index)

        # vids from cids 
        # vids = self.cids_to_vids(cids)
        vids = self.cid_to_first_vid[cids]

        # get weights 
        joint_ids1 = self.skinning_indices1[vids]
        weights1   = self.skinning_weights1[vids]
        joint_ids2 = self.skinning_indices2[vids]
        weights2   = self.skinning_weights2[vids]

        # lbsModel slow
        lbsModel = torch.zeros([len_col, len_cid, 4, 4], device=joint_ids1.device)
        for i in range(4):
            lbsModel += self.get_onedim_lbsModel(joint_ids1[..., i], weights1[..., i], batches, frames)
            lbsModel += self.get_onedim_lbsModel(joint_ids2[..., i], weights2[..., i], batches, frames)

        # position
        v_positions = self.v_position[vids]
        pos = torch.cat((v_positions, torch.ones(len_col, len_cid, 1, device=lbsModel.device)), dim=-1) 
        pos = pos.reshape(len_col, len_cid, 4).unsqueeze(-1)

        # get moved position
        fPosition = torch.matmul(lbsModel, pos).squeeze(-1)
        fPosition[negative_index] = torch.FloatTensor([0, 0, 0, 0]).unsqueeze(0).repeat(negative_index[0].shape[0], 1).to(self.device)
        
        return fPosition[..., :3]

    """ normal """
    def get_normal_from_vid(self, vids, batches, frames): # input [shape] -> output [shape, 4]  # (shape should be same)
        lbs_shape = (1,) * (len(vids.shape)) + (4, 4)
        lbsModel = torch.zeros_like(vids, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).repeat(*lbs_shape).to(vids.device)
        positions_shape = (1,) * (len(vids.shape)) + (3,)
        v_normals = torch.zeros_like(vids, dtype=torch.float).unsqueeze(-1).repeat(*positions_shape).to(vids.device)
        one_shape = (1,) * (len(vids.shape)) + (1,)
        one_tensor = torch.ones_like(vids, dtype=torch.float).unsqueeze(-1).repeat(*one_shape).to(vids.device)
        
        # get weights 
        joint_ids1 = self.skinning_indices1[vids]
        weights1   = self.skinning_weights1[vids]
        joint_ids2 = self.skinning_indices2[vids]
        weights2   = self.skinning_weights2[vids]

        # lbsModel [shape,4,4]
        lbsModel1 = torch.zeros_like(lbsModel, device=lbsModel.device)
        lbsModel2 = torch.zeros_like(lbsModel, device=lbsModel.device)
        for i in range(4):
            lbsModel1 += self.get_onedim_lbsModel(joint_ids1[..., i], weights1[..., i], batches, frames)
            lbsModel2 += self.get_onedim_lbsModel(joint_ids2[..., i], weights2[..., i], batches, frames)
        lbsModel = lbsModel1 + lbsModel2

        # self.v_normal [shape,3]
        uni_vals = torch.unique(vids)
        for val in uni_vals:
            vid_indices = torch.where(vids == val) #[0]
            v_normals[vid_indices] = self.v_normal[int(val)] # self.v_normal
        normal = torch.cat((v_normals, one_tensor), dim=-1).unsqueeze(-1)

        # get moved position
        fNormal = torch.matmul(lbsModel, normal).squeeze(-1).to(self.device)
        
        return fNormal[..., :3]

    """ cid from jid"""
        
    def get_vids_and_vpos_from_jids(self, batches, frames, jids): 
        len_jids = jids.shape[0]

        # jid -> cid and lbs 
        cid_max_len = max([len(self.perjoint_vids[jid]) for jid in jids])
        vids = torch.full((len(jids), cid_max_len), -1)
        for i in jids:
            vids[i, :len(self.perjoint_vids[i])] = torch.tensor(self.perjoint_vids[i])
        invalid_vids = torch.where(vids == -1)

        jids1_of_cids = self.skinning_indices1[vids]
        weights1_of_cids = self.skinning_weights1[vids]
        jids2_of_cids = self.skinning_indices2[vids]
        weights2_of_cids = self.skinning_weights2[vids]
        
        # lbsModel
        lbsModel = torch.zeros([len_jids, cid_max_len, 4, 4], device=jids.device)
        for i in range(4):
            lbsModel += self.get_onedim_lbsModel(jids1_of_cids[..., i], weights1_of_cids[..., i], batches.unsqueeze(-1).repeat(1, cid_max_len), frames.unsqueeze(-1).repeat(1, cid_max_len))
            lbsModel += self.get_onedim_lbsModel(jids2_of_cids[..., i], weights2_of_cids[..., i], batches.unsqueeze(-1).repeat(1, cid_max_len), frames.unsqueeze(-1).repeat(1, cid_max_len))

        # Get v position in T pose 
        v_positions = torch.zeros(len_jids, cid_max_len, 3).to(self.device)
        for i, jid in enumerate(jids):
            if self.perjoint_vposition[jid].shape[0] == 0:
                continue
            perjoint_vpos = self.perjoint_vposition[jid].to(self.device) 
            v_positions[i, :perjoint_vpos.shape[0]] = perjoint_vpos.to(self.device) 
        pos = torch.cat((v_positions, torch.ones(len_jids, cid_max_len, 1, device=jids.device)), dim=-1)
        pos = pos.unsqueeze(-1)

        # Update vertex position 
        fPosition = torch.matmul(lbsModel, pos).squeeze(-1)
        position = fPosition[..., :3]

        # exception handling
        position[invalid_vids] = torch.FloatTensor([0,0,0]).unsqueeze(0).repeat(invalid_vids[0].shape[0], 1).to(self.device)

        return vids, position 
    
    """ cid from fid """
    # aabbs : split into get positions from fids and get aabb from pos
    def get_cids_from_fids(self, fids): # fids[col_n, fids] -> aabb [n, cids, 3]
        cids = torch.full((fids.shape[0], fids.shape[1], 3), -1).to(self.device)

        # remove -1 
        index = torch.where(fids != -1)
        valid_fids = fids[index]
        
        # Cids from fids 
        fid_to_cids = self.fid_to_cids.to(cids.device)
        cids[index] = fid_to_cids[valid_fids]
        
        return cids 

    def get_cid_from_fid(self, fids): # fids[fids] -> aabb [cids, 3]
        # num_col = fids.shape[0]
        num_fids = fids.shape[0]
        cids = torch.full((num_fids, 3), -1).to(self.device)
        
        # remove -1 
        index = torch.where(fids != -1)
        valid_fids = fids[index]
        
        # Cids from fids 
        valid_cids = self.fid_to_cids[valid_fids]
        cids[index] = valid_cids
        
        return cids 

