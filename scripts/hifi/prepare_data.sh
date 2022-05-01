# $1 -> '/home/ubuntu/vakyansh-tts/data/ml_female/wav_22k' #give multiple folders separated by comma(,)
# $2 -> "female"
input_wav_path=$1
gender=$2

output_data_path='../../data/hifi/'$gender

valid_samples=1
test_samples=0

mkdir -p $output_data_path
python ../../utils/hifi/prepare_iitm_data_hifi.py -i $input_wav_path -v $valid_samples -t $test_samples -d $output_data_path
