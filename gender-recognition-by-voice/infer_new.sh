rm -rf new_results.txt

for FILE in ../data/iter_0/wav/*	  
do
    python3 test.py -f $FILE > temp_n.txt
    echo $FILE" "$( tail -1 temp_n.txt ) >> new_results.txt
done

rm temp_n.txt
