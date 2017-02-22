#encoding:utf-8

import json
import random
import datetime
from sys import argv
import matplotlib.pyplot as plt
import urllib


def add_dict_value(dic,k):
    if dic.get(k):
        dic[k] = dic[k] + 1
    else:
        dic[k] = 1
    return dic


def sortedDictValues2(adict):
    keys = adict.keys()
    keys.sort()
    return map(adict.get, keys)


def plotpic(sample_dict):
    x = range(len(sample_dict[0]))
    y = sample_dict[1]
    plt.plot(x, y, 'ro-')
    plt.xticks(x, sample_dict[0], rotation=45)
    plt.margins(0.08)
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def get_user_query_size(samplelist,deviceid):
    query_date = {}

    for n in samplelist:
        if n['deviceid'] == deviceid:
            add_dict_value(query_date,n['date'])

    return query_date



#sorted_data = sorted(data,key = lambda e:e.__getitem__('deviceid'))
result_sample = []
test_sample = []
sample_list = []
testsamplesum = {}
unsuccesssample = {}
fail = []
cancel = []
suggest = []
start_date = argv[1]
start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
data = []


#if the argv > 2 means we have a time duration
if len(argv)>2:
    end_date = argv[2]
    end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
    for i in range((end_date - start_date).days+1):
        date = start_date + datetime.timedelta(days=i)
        url = "http://ni.singulariti.io:32419/stats/dump-stats-by-day?start=" + date.strftime('%Y-%m-%d') + "&sig=qdjzni"
        page = urllib.urlopen(url)
        log = page.read()
        tempdata = json.loads(log)
        pre_day = {}
        for item in tempdata:
            if item['action'] == "action_perform_result":
                item['date'] = date.strftime('%Y-%m-%d')
                result_sample.append(item)
                testsamplesum = add_dict_value(testsamplesum,item['deviceid'])
        data.extend(tempdata)


else:
    url = "http://ni.singulariti.io:32419/stats/dump-stats-by-day?start=" + start_date.strftime('%Y-%m-%d') + "&sig=qdjzni"
    page = urllib.urlopen(url)
    log = page.read()
    data = json.loads(log)

    for item in data:
        if item['action'] == "action_perform_result":
            item['date'] = start_date.strftime('%Y-%m-%d')
            result_sample.append(item)
            testsamplesum = add_dict_value(testsamplesum,item['deviceid'])


for item in result_sample:
    if testsamplesum.get(item['deviceid']) < 50:
        sample_list.append(item)
    else:
        test_sample.append(item)


if len(test_sample) != 0:
    test_sample_size = 20
    if len(argv)>2:
        test_sample_size = test_sample_size * (end_date - start_date).days
    slice = random.sample(test_sample,test_sample_size)
    sample_list.extend(slice)

sample_list = sorted(sample_list,key = lambda e:e.__getitem__('deviceid'))

for item in sample_list:

    if item['kvs'].get('success') != "success":

        if item['kvs'].get('success') == "fail":
            fail.append(item)
        elif item['kvs'].get('success') == "cancel":
            cancel.append(item)
    elif item['kvs'].get('src') == "suggest":
        suggest.append(item)

print("user samples sum")
print(len(sample_list))
print("success rate")
print(100.0-((len(fail)+len(cancel)+len(suggest))/float(len(sample_list)))*100)
print("success rate(with suggest)")
print(100.0-((len(fail)+len(cancel))/float(len(sample_list)))*100)
for item in test_sample:
    if item['kvs'].get('success') != "success":

        if item['kvs'].get('success') == "fail":
            fail.append(item)
        elif item['kvs'].get('success') == "cancel":
            cancel.append(item)
    elif item['kvs'].get('src') == "suggest":
        suggest.append(item)


testsamplesum = sorted(testsamplesum.items(), key=lambda d: d[1])


sample_type = raw_input("sample type(fail,cancel,suggest) or deviceid, 'e' to exit:")



while sample_type != "e":
    if sample_type == "fail":
        print("fail samples")
        print(len(fail))
        for n in fail:
            print('%-50s%-30s' %(n['deviceid'],n['query']))
    if sample_type == "cancel":
        print("cancel samples")
        print(len(cancel))
        for n in cancel:
            print('%-50s%-30s' %(n['deviceid'],n['query']))
    if sample_type == "suggest":
        print("suggest samples")
        print(len(suggest))
        for n in suggest:
            print('%-50s%-30s' %(n['deviceid'],n['query']))
    if sample_type == "deviceid":
        for n,v in testsamplesum:
            print('%-50s%-30s' %(n,v))
        deviceid = raw_input("input deviceid, 'e' to back:")
        while deviceid != "e":

            query_date_list = get_user_query_size(result_sample,deviceid)
            if query_date_list:
                sorted_list = sorted(query_date_list.iteritems(), key=lambda d:d[0], reverse = False)
                query_date_list = zip(*sorted_list)

                plotpic(query_date_list)
            for n in result_sample:
                if n['deviceid'] == deviceid:
                    print('%-15s%-30s%-50s' %(n['date'],n['query'],n['kvs'].get('success')))
            deviceid = raw_input("input deviceid, 'e' to back:")


    sample_type = raw_input("sample type(fail,cancel,suggest) or deviceid, 'e' to exit:")
#x = range(1,100)
#plt.plot(x)
#plt.show()