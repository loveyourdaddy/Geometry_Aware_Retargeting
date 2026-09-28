# bash prepare/download_checkpoints.sh
FILE_ID="1dvfJW2fsTuskRZX7gOuwtDTLLZ0JOSW3"  # best.pt on Google Drive

mkdir -p saved
echo "Downloading checkpoint into saved/best.pt"
gdown "https://drive.google.com/uc?id=$FILE_ID" -O saved/best.pt
echo "Downloading done!"
