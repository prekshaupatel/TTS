file=$1
male=$(cat $file | awk -F ' ' '{print $4}' | awk -F '%' '{if ($1 > 70) print $1}' | wc -l)
female=$(cat $file | awk -F ' ' '{print $4}' | awk -F '%' '{if ($1 < 30) print $1}' | wc -l)
total=$(( male + female ))
m=$(echo $(( 10000 * $male / $total )) | sed -e 's/..$/.&/;t' -e 's/.$/.0&/')
f=$(echo $(( 10000 * $female / $total )) | sed -e 's/..$/.&/;t' -e 's/.$/.0&/')
echo "Male: "$m"%"
echo "Female: "$f"%"

