
cd /home/ubuntu/vakyansh-tts/checkpoints/glow
rm -f glow.zip

wget https://storage.googleapis.com/vakyansh-open-models/tts/malayalam/ml-IN/male_voice_1/glow.zip
unzip glow.zip

rm -rf male
mkdir male
mv glow/* male

rm -f glow.zip
rm -rf glow

cp male/config.json ../../config/glow/male.json

wget https://storage.googleapis.com/vakyansh-open-models/tts/malayalam/ml-IN/female_voice_0/glow.zip
unzip glow.zip

rm -rf female
mkdir female
mv fe_glow/* female

rm -f glow.zip
rm -rf fe_glow

cp female/config.json ../../config/glow/female.json


cd /home/ubuntu/vakyansh-tts/checkpoints/hifi

wget https://storage.googleapis.com/vakyansh-open-models/tts/malayalam/ml-IN/male_voice_1/hifi.zip
unzip hifi.zip

rm -rf male
mkdir male
mv hifi/* male

rm -f hifi.zip
rm -rf hifi

cd /home/ubuntu/vakyansh-tts/checkpoints/hifi

wget https://storage.googleapis.com/vakyansh-open-models/tts/malayalam/ml-IN/female_voice_0/hifi.zip
unzip hifi.zip

rm -rf female
mkdir female
mv hifi/* female

rm -f hifi.zip
rm -rf hifi

