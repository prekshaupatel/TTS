# $1 -> '/home/ubuntu/vakyansh-tts/data/iter_0/line_male_index.txt'
# $2 -> '/home/ubuntu/vakyansh-tts/data/iter_0/male_wav_22k'
# $3 -> 'male'

input_text_path=$1
input_wav_path=$2
gender=$3


output_data_path='../../data/glow/'$gender

valid_samples=0
test_samples=0

mkdir -p $output_data_path
python ../../utils/glow/prepare_iitm_data_glow_en.py -i $input_text_path -o $output_data_path -w $input_wav_path -v $valid_samples -t $test_samples
