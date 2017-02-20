import json
import random

f = open("data.json")
data = json.load(f)
f.close()

#sorted_data = sorted(data,key = lambda e:e.__getitem__('deviceid'))
result_sample = []
test_sample = []
sample_list = []
testsamplesum = {}
unsuccesssample = {}
fail = []
cancel = []
suggest = []
for item in data:
    if item['action'] == "action_perform_result":
        result_sample.append(item)
        if testsamplesum.get(item['deviceid']):
            testsamplesum[item['deviceid']] = testsamplesum[item['deviceid']] + 1
        else:
            testsamplesum[item['deviceid']] = 1
for item in result_sample:
    if testsamplesum.get(item['deviceid']) < 50:
        sample_list.append(item)
    else:
        test_sample.append(item)
if len(test_sample) != 0:
    slice = random.sample(test_sample,20)
    sample_list.extend(slice)

print(len(sample_list))

for item in sample_list:

    if item['kvs'].get('success') != "success":

        if item['kvs'].get('success') == "fail":
            fail.append(item['query'])
        elif item['kvs'].get('success') == "cancel":
            cancel.append(item['query'])
    elif item['kvs'].get('src') == "suggest":
        suggest.append(item['query'])

print("user samples sum")
print(len(sample_list))
print("fail samples")
print(len(fail))
for n in fail:
    print(n)
print("cancel samples")
print(len(cancel))
for n in cancel:
    print(n)
print("suggest samples")
print(len(suggest))
for n in suggest:
    print(n)
print("success rate")
print(100.0-((len(fail)+len(cancel)+len(suggest))/float(len(sample_list)))*100)
print("success rate(with suggest)")
print(100.0-((len(fail)+len(cancel))/float(len(sample_list)))*100)