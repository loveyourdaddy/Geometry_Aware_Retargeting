import torch
import pickle

""" Render setting """
def render_motions(args, characters, motions):
    # Arrange
    distance = 2.0
    for i in range(len(characters)):
        for f in range(len(motions[i].poses)):
            motions[i].poses[f].translate_root_p([i*distance, 0, 0])




def render_result(args,
                  source0_character, source1_character, target0_character, target1_character,
                  source0_motion, source1_motion, output0, output1, align_z_direction=False):
    # Arrange
    distance = 4.0 # 5 7
    for f in range(len(source1_motion.poses)):
        # output
        # if align_z_direction: 
        #     # z direction
        #     # position       = glm.vec3(5, 1.2, 0), # from +x
        #     source0_motion.poses[f].translate_root_p([0, 0, distance/2])
        #     source1_motion.poses[f].translate_root_p([0, 0, distance/2])
        #     output0.poses[f].translate_root_p([0, 0, -distance/2]) 
        #     output1.poses[f].translate_root_p([0, 0, -distance/2])   
        # else: 
        #     # x direction
        #     # position       = glm.vec3(0, 1.2, 5), # +z 
        #     if args.motion0 == "move_03_03_male_30fps":
        #         # target -x align for front view
        #         source0_motion.poses[f].translate_root_p([distance/2, 0, 0])
        #         source1_motion.poses[f].translate_root_p([distance/2, 0, 0])
        #         output0.poses[f].translate_root_p([-distance/2, 0, 0])
        #         output1.poses[f].translate_root_p([-distance/2, 0, 0])
        #     else: 
        #         # target +x align for front view
        #         source0_motion.poses[f].translate_root_p([-distance/2, 0, 0])
        #         source1_motion.poses[f].translate_root_p([-distance/2, 0, 0])
        #         output0.poses[f].translate_root_p([distance/2, 0, 0])
        #         output1.poses[f].translate_root_p([distance/2, 0, 0])
        source0_motion.poses[f].translate_root_p([-distance, 0, 0]) 
        source1_motion.poses[f].translate_root_p([-distance, 0, 0])   
        
    characters = []
    characters.append(source0_character)  # source
    characters.append(source1_character)
    characters.append(target0_character)  # output
    characters.append(target1_character)

    motions = []
    motions.append(source0_motion)  # source
    motions.append(source1_motion)
    motions.append(output0)  # output
    motions.append(output1)

    return characters, motions

def render_compare(args,
                  source0_character, source1_character, target0_character, target1_character, compare0_character, compare1_character,
                  source0_motion, source1_motion, output0, output1, compare_motion0, compare_motion1):
    # Arrange
    distance = 4.0 # 5 7
    for f in range(len(source1_motion.poses)):
        # source
        source0_motion.poses[f].translate_root_p([args.source_pos, 0, 0])
        source1_motion.poses[f].translate_root_p([args.source_pos, 0, 0])
        # output
        output0.poses[f].translate_root_p([args.source_pos+distance, 0, 0])
        output1.poses[f].translate_root_p([args.source_pos+distance, 0, 0])
        # compare
        compare_motion0.poses[f].translate_root_p([args.source_pos+2*distance, 0, 0])
        compare_motion1.poses[f].translate_root_p([args.source_pos+2*distance, 0, 0])

    characters = []
    characters.append(source0_character)  # source
    characters.append(source1_character)
    characters.append(target0_character)  # output
    characters.append(target1_character)
    characters.append(compare0_character)  # output
    characters.append(compare1_character)

    motions = []
    motions.append(source0_motion)  # source
    motions.append(source1_motion)
    motions.append(output0)  # output
    motions.append(output1)
    motions.append(compare_motion0)  # compare
    motions.append(compare_motion1)

    return characters, motions

def render_two_motion(args,
                      character0, character1,
                      motion0, motion1):

    # for f in range(len(motion0.poses)):
    #     motion0.poses[f].translate_root_p([args.source_pos, 0, 0])
    #     motion1.poses[f].translate_root_p([args.joint_pos, 0, 0])

    # source joint geo
    characters = []
    characters.append(character0)
    characters.append(character1)
    # characters.append(character2)

    motions = []
    motions.append(motion0)
    motions.append(motion1)
    # motions.append(motion2)

    return characters, motions


def render_one_motion(args,
                      character0,
                      motion0):

    # source joint geo
    characters = []
    characters.append(character0)
    # characters.append(character1)

    motions = []
    motions.append(motion0)
    # motions.append(motion1)

    return characters, motions


def set_motion(character0, character1,
               motion0, motion1):

    # source joint geo
    characters = []
    characters.append(character0)
    # characters.append(character1)
    # characters.append(character2)

    motions = []
    motions.append(motion0)
    # motions.append(motion1)
    # motions.append(motion2)

    return characters, motions

""" useful functions """


def update_index(target, source, index):
    target_copy = torch.clone(target)
    target_copy[index] = source
    return target_copy


def remove_negative_index(joint_ids, negative_idx):
    return update_index(joint_ids, 0, negative_idx)


def deepcopy(character):
    return pickle.loads(pickle.dumps(character, -1))
