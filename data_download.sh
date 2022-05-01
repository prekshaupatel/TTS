
cd /home/ubuntu/vakyansh-tts

cd data
mkdir ml_male

cd ml_male
mkdir wav

wget https://www.openslr.org/resources/63/ml_in_male.zip
unzip -o ml_in_male.zip
rm -f ml_in_male.zip

mv *.wav wav
rm -f line_index.tsv

wget https://www.openslr.org/resources/63/line_index_male.tsv

awk -F '\t' '{printf "( %s \"%s\" )\n", $1, $2}' line_index_male.tsv > line_index.txt

rm -f line_index_male.tsv



cd /home/ubuntu/vakyansh-tts

cd data
mkdir ml_female

cd ml_female
mkdir wav

wget https://www.openslr.org/resources/63/ml_in_female.zip
unzip -o ml_in_female.zip
rm -f ml_in_female.zip

mv *.wav wav
rm -f line_index.tsv

wget https://www.openslr.org/resources/63/line_index_female.tsv

awk -F '\t' '{printf "( %s \"%s\" )\n", $1, $2}' line_index_female.tsv > line_index.txt

rm -f line_index_female.tsv
