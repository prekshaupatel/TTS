rm -rf female_results.txt

i=0
for FILE in ../vakyansh-tts/data/ml_female/wav/*	  
do
    ((i++))
    if [[ $(( $i % 20 )) == 0 ]]
    then
       python3 test.py -f $FILE > temp_f.txt
       echo $FILE" "$( tail -1 temp_f.txt ) >> female_results.txt
    fi
done

rm temp_f.txt
