gender='male'
glowdir='../../checkpoints/glow/'$gender'/'
hifidir='../../checkpoints/hifi/'$gender'/'
device='gpu'
lang='ml'


python ../../utils/inference/run_gradio.py -a $glowdir -v $hifidir -d $device -L $lang
