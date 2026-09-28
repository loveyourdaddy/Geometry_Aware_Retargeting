# bash prepare/download_data.sh
FILE_ID="1ICs-MCKx0roilw2mU78wRCpKbpeLHKtJ"  # motions.zip on Google Drive

mkdir -p ../Resource
echo "Downloading motions into ../Resource/"
gdown "https://drive.google.com/uc?id=$FILE_ID" -O motions.zip
unzip -n -q motions.zip -d ../Resource
rm motions.zip
echo "Downloading done!"
