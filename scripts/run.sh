gender='male'

index_file='/home/ubuntu/vakyansh-tts/data/iter_0/line_'$gender'_index.txt' 
wav_folder='/home/ubuntu/vakyansh-tts/data/iter_0/'$gender'_wav_22k'


# EDIT ~/vakyansh-tts/checkpoints/glow/$gender/config.json

cd ~/vakyansh-tts/scripts/glow

bash prepare_data.sh $index_file $wav_folder $gender
bash train_glow.sh $gender


cd ~/vakyansh-tts/scripts/hifi

bash prepare_data.sh $wav_folder $gender
bash train_hifi.sh $gender 60000


cd ~/vakyansh-tts/scripts/inference

bash test_score.sh $gender
