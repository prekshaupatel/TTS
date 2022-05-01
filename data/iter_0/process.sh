cd ~/vakyansh-tts/gender-recognition-by-voice
# bash infer_new.sh
bash gen_diff_files.sh

cd ~/vakyansh-tts/data/iter_0

sort -u ~/gender-recognition-by-voice/new_female_list.txt > new_female_list.txt
sort -u ~/gender-recognition-by-voice/new_male_list.txt > new_male_list.txt

mv output_mono_baseline.tsv output_asr_iteration_0.tsv

sort -u output_asr_iteration_0.tsv > temp.tsv
mv temp.tsv output_asr_iteration_0.tsv

join -t $'\t' new_male_list.txt output_asr_iteration_0.tsv > male_list.tsv
join -t $'\t' new_female_list.txt output_asr_iteration_0.tsv > female_list.tsv

awk -F '[\t\.]' '{printf "( %s \"%s\" )\n", $1, $3}' male_list.tsv > line_male_index.txt
awk -F '[\t\.]' '{printf "( %s \"%s\" )\n", $1, $3}' female_list.tsv > line_female_index.txt

mkdir male_wav
while read FILE;
do
    mv -f 'wav/'$FILE male_wav
done < new_male_list.txt

mkdir female_wav
while read FILE;
do
    mv -f 'wav/'$FILE female_wav
done < new_female_list.txt

echo 'Male'
wc -l line_male_index.txt
ls male_wav | wc -l

echo 'Female'
wc -l line_female_index.txt
ls female_wav | wc -l

rm -rf male_wav_22k
rm -rf female_wav_22k

cd ~/vakyansh-tts/scripts/data

input_wav_path='/home/ubuntu/vakyansh-tts/data/iter_0/male_wav/'
output_wav_path='/home/ubuntu/vakyansh-tts/data/iter_0/male_wav_22k/'

bash resample.sh $input_wav_path $output_wav_path

input_wav_path='/home/ubuntu/vakyansh-tts/data/iter_0/female_wav/'
output_wav_path='/home/ubuntu/vakyansh-tts/data/iter_0/female_wav_22k/'

bash resample.sh $input_wav_path $output_wav_path


cd ~/vakyansh-tts/data/iter_0

cat ../ml_male/line_index_train.txt line_male_index.txt > temp.txt
mv temp.txt line_male_index.txt

cp ../ml_male/wav_22k/* male_wav_22k

cat ../ml_female/line_index_train.txt line_female_index.txt > temp.txt
mv temp.txt line_female_index.txt

cp ../ml_female/wav_22k/* female_wav_22k

echo 'Male'
wc -l line_male_index.txt
ls male_wav_22k | wc -l

echo 'Female'
wc -l line_female_index.txt
ls female_wav_22k | wc -l

# rm -f new_female_list.txt new_male_list.txt output_asr_iteration_0.tsv male_list.tsv female_list.tsv
