rm -rf male_results.txt

i=0
for FILE in ../vakyansh-tts/data/ml_male/wav/*	  
do
    ((i++))
    if [[ $(( $i % 20 )) == 0 ]]
    then
       python3 test.py -f $FILE > temp.txt
       echo $FILE" "$( tail -1 temp.txt ) >> male_results.txt
    fi
done

rm temp.txt
