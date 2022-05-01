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

rm -f label_$gender.tsv
touch label_$gender.tsv
while read text;
do
    echo "$text";
    timestamp=$(($timestamp+1))
    wav_file=$wav'/wav/'$timestamp'.wav'
    python ../../utils/inference/tts.py -a $glowdir -v $hifidir -d $device -t "$text" -w $wav_file 
    echo "File saved at: "$wav_file
    echo -e $timestamp'.wav\t'$text >> label_$gender.tsv
done < cleaned_text.txt
rm cleaned_text.txt

mv label_$gender.tsv $wav

cd ../../results
zip -r $dataset'_'$gender'.zip' $dataset'_'$gender
rm -rf $dataset'_'$gender


