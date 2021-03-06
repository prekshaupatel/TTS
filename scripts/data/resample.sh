input_wav_path=$1 # '/home/ubuntu/vakyansh-tts/data/iter_0/male_wav/'
output_wav_path=$2 # '/home/ubuntu/vakyansh-tts/data/iter_0/male_wav_22k/'
output_sample_rate=22050

#######################

dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

mkdir -p $output_wav_path
python $parentdir/utils/data/resample.py -i $input_wav_path -o $output_wav_path -s $output_sample_rate

python $parentdir/utils/data/duration.py $output_wav_path
