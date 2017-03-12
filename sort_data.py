#encoding:utf-8

import json
import random
import datetime
from sys import argv
import matplotlib.pyplot as plt
import urllib
import collections

def print_info_by_type(result_type):
    print("samples size:")
    print(len(result_type))
    for n in result_type:
        print('%-50s%-30s' %(n['deviceid'],n['query']))

def plotpic(sample_dict):
    x = range(len(sample_dict[0]))
    y = sample_dict[1]
    plt.plot(x, y, 'ro-')
    plt.xticks(x, sample_dict[0], rotation=45)
    plt.margins(0.08)
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def get_subset_by_result(samples):
    fail = []
    cancel = []
    suggest = []
    recommend = []
    for item in samples:
        if item['kvs'].get('success') != "success":
            if item['kvs'].get('success') == "fail":
                fail.append(item)
            elif item['kvs'].get('success') == "cancel":
                cancel.append(item)
        elif item['kvs'].get('src') == "suggest":
            suggest.append(item)
        elif item['kvs'].get('src') == "recommend":
            recommend.append(item)
    return fail,cancel,suggest,recommend

def get_sample_by_date(date):
    result_sample = []
    url = "http://ni.singulariti.io:32419/stats/dump-stats-by-day?start=" + date.strftime('%Y-%m-%d') + "&sig=qdjzni"
    page = urllib.urlopen(url)
    log = page.read()
    data = json.loads(log)
    for item in data:
        if item['action'] == "action_perform_result":
            item['date'] = date.strftime('%Y-%m-%d')
            result_sample.append(item)
    return result_sample

test_sample = []
sample_list = []
start_date = argv[1]
start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
result_sample = []

#if the argv > 2 means we have a time duration
if len(argv)>2:
    end_date = argv[2]
    end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
    for i in range((end_date - start_date).days+1):
        date = start_date + datetime.timedelta(days=i)
        result_sample.extend(get_sample_by_date((date)))
else:
    result_sample = get_sample_by_date(start_date)

test_list = [item['deviceid'] for item in result_sample]
testsamplesum = collections.Counter(test_list)

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

fail_list,cancel_list,suggest_list,recommend_list = get_subset_by_result(sample_list)

print("user samples sum")
print(len(sample_list))
print("success rate")
print(100.0-((len(fail_list)+len(cancel_list)+len(suggest_list)+len(recommend_list))/float(len(sample_list)))*100)
print("success rate(with suggest)")
print(100.0-((len(fail_list)+len(cancel_list)+len(recommend_list))/float(len(sample_list)))*100)

fail,cancel,suggest,recommend = get_subset_by_result(test_sample)
fail_list.extend(fail)
cancel_list.extend(cancel)
suggest_list.extend(suggest)
recommend_list.extend(recommend)

testsamplesum = sorted(testsamplesum.items(), key=lambda d: d[1])

sample_type = raw_input("sample type(fail,cancel,suggest,recommend) or deviceid, 'e' to exit:")

while sample_type != "e":
    if sample_type == "fail":
        print_info_by_type(fail_list)
    if sample_type == "cancel":
        print_info_by_type(cancel_list)
    if sample_type == "recommend":
        print_info_by_type(recommend_list)
    if sample_type == "suggest":
        print_info_by_type(suggest_list)
    if sample_type == "deviceid":
        for n,v in testsamplesum:
            print('%-50s%-30s' %(n,v))
        deviceid = raw_input("input deviceid, 'e' to back:")
        while deviceid != "e":

            test_list = [item['date'] for item in result_sample if item['deviceid']==deviceid]
            query_date_list = collections.Counter(test_list)
            if query_date_list:
                sorted_list = sorted(query_date_list.iteritems(), key=lambda d:d[0], reverse = False)
                query_date_list = zip(*sorted_list)
                plotpic(query_date_list)
            for n in result_sample:
                if n['deviceid'] == deviceid:
                    print('%-15s%-30s%-50s' %(n['date'],n['query'],n['kvs'].get('success')))
            deviceid = raw_input("input deviceid, 'e' to back:")

    sample_type = raw_input("sample type(fail,cancel,suggest,recommend) or deviceid, 'e' to exit:")
