gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='cuda'
text='കൂടുതൽ വിവരങ്ങൾ വരുമ്പോൾ തിരിച്ചു ചേർക്കാം'

noise_scale='0.0'
length_scale='1.0'
transliteration=0
number_conversion=1
split_sentences=1
lang='ml'


timestamp=$(date +%s)
wav='../../results/'$gender'/'
wav_file=$wav/$timestamp'.wav'


mkdir -p $wav

python ../../utils/inference/advanced_tts.py -a $glowdir -v $hifidir -d $device -t "$text" -w $wav_file -L $lang -n $noise_scale -l $length_scale -T $transliteration -N $number_conversion -S $split_sentences
echo "File saved at: "$wav_file
