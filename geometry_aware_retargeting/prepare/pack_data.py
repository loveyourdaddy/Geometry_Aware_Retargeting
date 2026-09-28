'''
Build prepare/motions.zip for release (the file to upload to Google Drive)
    python prepare/pack_data.py
'''
import os
import re
import zipfile

RESOURCE = "../Resource"
SOURCE_CHARACTERS = ["SMPLx", "Ybot"]  # characters that have input motions
TPOSE_CHARACTERS = ["SMPLx", "SMPLx_fat", "Ybot", "Leonard", "Amy", "Ortiz"]
OUTPUT = "prepare/motions.zip"


def listed_motion_names():
    """All motion names written in option_motion.py (including commented-out ones)."""
    source = open("option_motion.py").read()
    pairs = re.findall(r'"(\w+)"\s*:\s*"(\w+)"', source)
    return sorted({name for pair in pairs for name in pair})


def main():
    paths = ["Tpose_template.bvh"]
    paths += ["motions/single_motion/{}/Tpose.bvh".format(c) for c in TPOSE_CHARACTERS]
    for character in SOURCE_CHARACTERS:
        for name in listed_motion_names():
            path = "motions/interaction_motion/{}/{}.bvh".format(character, name)
            if os.path.exists(os.path.join(RESOURCE, path)):
                paths.append(path)

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as f:
        for path in paths:
            f.write(os.path.join(RESOURCE, path), path)

    print("{}: {} files, {:.1f} MB".format(OUTPUT, len(paths), os.path.getsize(OUTPUT) / 1e6))


if __name__ == "__main__":
    main()
