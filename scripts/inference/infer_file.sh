gender='male'
dataset='kaggle'

glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cuda'

python3 clean.py mono_$dataset.txt cleaned_text.txt

timestamp=$(date +%s)
wav='../../results/'$dataset'_'$gender

rm -rf $wav
mkdir -p $wav/wav

python ../../utils/inference/tts_file.py -a $glowdir -v $hifidir -d $device -w $wav -s $timestamp -f "label_"$gender".tsv" -t "cleaned_text.txt"

rm cleaned_text.txt

cd ../../results
zip -r $dataset'_'$gender'.zip' $dataset'_'$gender
rm -rf $dataset'_'$gender


