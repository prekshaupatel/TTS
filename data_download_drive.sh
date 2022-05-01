
cd /home/ubuntu/vakyansh-tts

cd data
mkdir iter_0

cd iter_0
mkdir wav

gdown 'https://drive.google.com/uc?id=1rmSSCNvmbleZEfL9EW3Y9DRV3-keY0rL'
gdown 'https://drive.google.com/uc?id=1N5lMsjoUCuFeo3G2ba4j2XR-cy-m99V_'

unzip -o monolingual_audio.zip
rm -f monolingual_audio.zip

rm -rf __MACOSX
mv m*/*.wav wav

rm -rf monolingual_audio

awk -F '[\t\.]' '{printf "( %s \"%s\" )\n", $1, $3}' output_asr_iteration_0.tsv > line_index.txt

# rm -f output_asr_iteration_0.tsv

