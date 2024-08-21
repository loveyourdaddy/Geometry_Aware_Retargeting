# target0 (ptn) vs target1 (deformed)
trainset_bvh = {
    # captured (9)
    "HandShaking_JIS": "HandShaking_SCW",
    "HuggingNormal_SCW": "HuggingNormal_JIS",
    "HuggingCross_SCW": "HuggingCross_JIS",
    "HipTouch_JIS": "HipTouch_SCW",
    "Rapperstylegreeting_JIS": "Rapperstylegreeting_SCW",

    # CMU (4)
    "Easy_01_male_30fps": "Easy_01_female_30fps",
    "Medium_03_male_30fps": "Medium_03_female_30fps",
    "Hard_04_male_30fps": "Hard_04_female_30fps",
    "move_03_03_male_30fps":"move_03_03_female_30fps",

    "push_up001_S1":"push_up001_S2",
    "greeting002_S1":"greeting002_S2",
    "jump_crawl001_S1":"jump_crawl001_S2",
    "one_leg_back_stretch_S1":"one_leg_back_stretch_S2",
    # "stack_push_up001_S1":"stack_push_up001_S2",
    
    # auramesh
    "martial_SpinTurn_A": "martial_SpinTurn_B",
    "neckback_A": "neckback_B",
    
    # some mirror
    # captured
    "mirror_HandShaking_JIS":"mirror_HandShaking_SCW",
    "mirror_HandShaking_Side_JIS":"mirror_HandShaking_Side_SCW",
    "mirror_HandShaking_Upper_JIS":"mirror_HandShaking_Upper_SCW",
    "mirror_HipTouch_JIS":"mirror_HipTouch_SCW",
    "mirror_Punch_JIS":"mirror_Punch_SCW",
    "mirror_Rapperstylegreeting_JIS":"mirror_Rapperstylegreeting_SCW",
    # CMU (4)
    "mirror_Easy_01_male_30fps": "mirror_Easy_01_female_30fps",
    "mirror_Medium_03_male_30fps": "mirror_Medium_03_female_30fps",
    "mirror_Hard_04_male_30fps": "mirror_Hard_04_female_30fps",
}

# target0 (ptn) vs target1 (main, deformed)
example_bvh = {
    # "greeting002_S1":"greeting002_S2",
    # "HandShaking_JIS": "HandShaking_SCW",
    # "Medium_03_male_30fps": "Medium_03_female_30fps",
    # "martial_SpinTurn_A": "martial_SpinTurn_B",
    # "neckback_A": "neckback_B",
    # "move_03_03_male_30fps":"move_03_03_female_30fps",
    # "jump_crawl001_S1":"jump_crawl001_S2",
    # "push_up001_S1":"push_up001_S2",
    # "stack_push_up001_S1":"stack_push_up001_S2",
    "one_leg_back_stretch_S1":"one_leg_back_stretch_S2",
    
    # captured
    # "HuggingNormal_SCW": "HuggingNormal_JIS",
    # "HuggingCross_SCW": "HuggingCross_JIS",
    # "HipTouch_JIS": "HipTouch_SCW",
    # "Rapperstylegreeting_JIS": "Rapperstylegreeting_SCW",

    # # CMU
    # "Easy_01_male_30fps": "Easy_01_female_30fps",
    # "Hard_04_male_30fps":"Hard_04_female_30fps",

    # seen mirror 
    # captured 
    # "mirror_HandShaking_JIS":"mirror_HandShaking_SCW",
    # "mirror_HandShaking_Side_JIS":"mirror_HandShaking_Side_SCW",
    # "mirror_HandShaking_Upper_JIS":"mirror_HandShaking_Upper_SCW",
    # "mirror_HipTouch_JIS":"mirror_HipTouch_SCW",
    # "mirror_Punch_JIS":"mirror_Punch_SCW",
    # "mirror_Rapperstylegreeting_JIS":"mirror_Rapperstylegreeting_SCW",
    # "mirror_HandShaking_JIS": "mirror_HandShaking_SCW",
    # CMU 
    # "mirror_Easy_01_male_30fps": "mirror_Easy_01_female_30fps",
    # "mirror_Medium_03_male_30fps": "mirror_Medium_03_female_30fps",
    # "mirror_Hard_04_male_30fps": "mirror_Hard_04_female_30fps",
    
    
    # unseen 
    # hug 
    # "hugging_A":"hugging_B",
    # "HuggingNormal_JIS": "HuggingNormal_SCW",
    # "Hugging_Normal003_JIS": "Hugging_Normal003_SCW",
    # "Hugging_Cross001_JIS": "Hugging_Cross001_SCW",

    # CMU
    # "High_five_S1":"High_five_S2",
    # "Easy_02_male_30fps":"Easy_02_female_30fps",
    # "Easy_03_male_30fps":"Easy_03_female_30fps",
    # "Medium_01_male_30fps":"Medium_01_female_30fps",
    # "Hard_05_male_30fps":"Hard_05_female_30fps", 
    # "move_02_02_male_30fps":"move_02_02_female_30fps",
    
    # HAC (8) # A1.1 Left Punch + Avoid (95개)
    # "SMPLx_1_C0": "SMPLx_1_C1",
    # "SMPLx_2_C0": "SMPLx_2_C1",
    # "SMPLx_4_C0": "SMPLx_4_C1",
    # "SMPLx_5_C0": "SMPLx_5_C1",
    # "SMPLx_6_C0": "SMPLx_6_C1",
    # "SMPLx_7_C0": "SMPLx_7_C1",
    # "SMPLx_8_C0": "SMPLx_8_C1",
    # "SMPLx_9_C0": "SMPLx_9_C1",
    
    # unseen mirror
    # "mirror_stack_push_up001_S1":"mirror_stack_push_up001_S2",
    # "mirror_push_up001_S1":"mirror_push_up001_S2",
    # "mirror_jump_crawl001_S1":"mirror_jump_crawl001_S2", # "jump_crawl001_S2":"jump_crawl001_S1",
    # "mirror_one_leg_back_stretch_S1":"mirror_one_leg_back_stretch_S2",
    # "mirror_High_five_S1":"mirror_High_five_S2",
    # hug
    # "mirror_hugging_A":"mirror_hugging_B",
    # "mirror_HuggingCross_JIS":"mirror_HuggingCross_SCW",
    # "mirror_HuggingNormal_JIS":"mirror_HuggingNormal_SCW",
    # auramesh
    # "mirror_martial_SpinTurn_A": "mirror_martial_SpinTurn_B",
    
    # "mirror_greeting002_S1":"mirror_greeting002_S2",
    # "mirror_move_03_03_male_30fps":"mirror_move_03_03_female_30fps", 
    # "mirror_neckback_A": "mirror_neckback_B",
    
    # limitation
    # "back_lift001_S2":"back_lift001_S1",
    # "mirror_back_lift001_S1":"mirror_back_lift001_S2",
}

""" RD """
# testset 모션(unseen)은 제외해주기
RD_bvh = {
    # captured (9)
    "HandShaking_JIS": "HandShaking_SCW",  # 1 hand shaking
    "HuggingNormal_SCW": "HuggingNormal_JIS",
    "HuggingCross_SCW": "HuggingCross_JIS",
    "HipTouch_JIS": "HipTouch_SCW",  # 4 Hiptouch_JIS
    "Rapperstylegreeting_JIS": "Rapperstylegreeting_SCW",  # 5 Rapperstylegreeting

    # CMU (4)
    "Easy_01_male_30fps": "Easy_01_female_30fps",
    "Medium_03_male_30fps": "Medium_03_female_30fps",
    "Hard_04_male_30fps": "Hard_04_female_30fps",
    "move_03_03_male_30fps":"move_03_03_female_30fps",

    "push_up001_S1":"push_up001_S2",
    "greeting002_S1":"greeting002_S2",
    "jump_crawl001_S1":"jump_crawl001_S2",
    "one_leg_back_stretch_S1":"one_leg_back_stretch_S2",
    "stack_push_up001_S1":"stack_push_up001_S2",
    
    # auramesh
    "martial_SpinTurn_A": "martial_SpinTurn_B",
    "neckback_A": "neckback_B",
    
    # some mirror
    # captured
    "mirror_HandShaking_JIS":"mirror_HandShaking_SCW",
    "mirror_HandShaking_Side_JIS":"mirror_HandShaking_Side_SCW",
    "mirror_HandShaking_Upper_JIS":"mirror_HandShaking_Upper_SCW",
    "mirror_HipTouch_JIS":"mirror_HipTouch_SCW",
    "mirror_Punch_JIS":"mirror_Punch_SCW",
    "mirror_Rapperstylegreeting_JIS":"mirror_Rapperstylegreeting_SCW",
    # CMU (4)
    "mirror_Easy_01_male_30fps": "mirror_Easy_01_female_30fps",
    "mirror_Medium_03_male_30fps": "mirror_Medium_03_female_30fps",
    "mirror_Hard_04_male_30fps": "mirror_Hard_04_female_30fps",
}

""" dfm joints """
# root 
ptn_root_motion = [
    # CMU
    "greeting002_S1",
    "jump_crawl001_S1",
    "move_03_03_male_30fps",
    "Hard_04_male_30fps",
     # aura mesh
    "martial_SpinTurn_A",
    "neckback_A",
]
dfm_root_motion = [
    # CMU
    "jump_crawl001_S1",
]
# 예외처리: descriptor을 특정 joint로 제한하고 싶을때
root_motion_by_root_vids = [
    "greeting002_S1",
]

# spine 
ptn_spine_motion = [
]
# exception for fat
ptn_spine_for_fat = [
    "move_03_03_male_30fps",
]
dfm_spine_motion = [
]

# leg 
ptn_leg_motion = [
    # CMU
    "push_up001_S1", 
    # "greeting002_S1",
    "one_leg_back_stretch_S1", 
    "stack_push_up001_S1",
    "neckback_A",
]
# "back_lift001_S1", 
dfm_leg_motion = [
    # CMU
    "push_up001_S1", 
    # "greeting002_S1",
    "jump_crawl001_S1",
    "one_leg_back_stretch_S1", 
    "stack_push_up001_S1",
]
# "back_lift001_S1", 

""" pene """
ptn_not_arm_motion = [
    "jump_crawl001_S1",
] 

""" start/end frame clampping """
start_frame_dict = {
    "HandShaking_JIS": 121,
    "HandShaking_Side_JIS": 138,
    "HipTouch_JIS": 126,
    "Rapperstylegreeting_JIS": 20,

    "Easy_01_male_30fps": 66,
    "Medium_01_male_30fps": 65,
    "Hard_04_male_30fps": 48,
    "greeting002_S1": 60,
    "one_leg_back_stretch_S1": 233,
}
end_frame_dict = {
    "HandShaking_JIS": 195,
    "HandShaking_Side_JIS": 196,
    "HandShaking_Upper_JIS": 212,
    "HipTouch_JIS": 150,
    "Rapperstylegreeting_JIS": 190,
    "Salsa_JIS": 519,

    "Easy_01_male_30fps": 1361,
    "Medium_01_male_30fps": 1516,
    "Hard_04_male_30fps": 1600,
    "move_02_02_male_30fps": 1160,
    
    "greeting002_S1": 320,
    "one_leg_back_stretch_S1": 650,
}

# align_motion_in_z_axis
align_motion_in_z_axis = [
    "greeting002_S1",
    # "push_up001_S1", 
    # "stack_push_up001_S1", 
    # "one_leg_back_stretch_S1"
    ]