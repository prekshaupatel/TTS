# $1 -> gender

gender=$1

glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cuda'

# python3 clean.py mono_$dataset.txt cleaned_text.txt

timestamp=$(date +%s)
wav='../../results/test/'$gender

cp ~/vakyansh-tts/data/ml_$gender/line_index_test.txt temp_text.txt
awk -F '"' '{print $2}' temp_text.txt > cleaned_text.txt

awk -F ' ' '{print $2}' temp_text.txt > ids.txt

rm -rf $wav
mkdir -p $wav/wav

python ../../utils/inference/tts_file_test.py -a $glowdir -v $hifidir -d $device -w $wav -s $timestamp -f "label_"$gender".tsv" -t "cleaned_text.txt"

rm cleaned_text.txt

awk -F '\t' '{print $1}' $wav/"label_"$gender".tsv" > ids_gen.txt

paste ids.txt ids_gen.txt > ids_merged.txt
rm ids.txt ids_gen.txt

python3 data_prep_2.py --org "../../data/ml_"$gender"/wav_22k_test" --gen $wav'/wav' --file "ids_merged.txt"
# python3 data_prep.py --org "../../data/ml_"$gender"/wav_22k_test" --gen $wav'/wav' --file "ids_merged.txt"
