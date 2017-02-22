#encoding:utf-8

import json
import datetime
from sys import argv
import matplotlib.pyplot as plt
import urllib
import datetime
duration = 7
version = "10042"

def plotpic(x_dim,y_dim,labelstr):
    x = range(len(x_dim))
    y = y_dim
    plt.plot(x, y, label = labelstr)
    plt.xticks(x, x_dim, rotation=45)
    plt.margins(0.08)
    plt.subplots_adjust(bottom=0.15)


def flatten_list(nested):
    if isinstance(nested, list):
        for sublist in nested:
            for item in flatten_list(sublist):
                yield item
    else:
        yield nested


def add_dict_value(dic,k):
    if dic.get(k):
        dic[k] = dic[k] + 1
    else:
        dic[k] = 1
    return dic


def get_user_list_by_date(start_date):
    pre_user = []
    new_user_num = []
    dates = []
    end_date = start_date + datetime.timedelta(days=duration+7)
    sumlist = []
    for i in range((end_date - start_date).days+1):
        date = start_date + datetime.timedelta(days=i)
        url = "http://ni.singulariti.io:32419/stats/dump-stats-by-day?start=" + date.strftime('%Y-%m-%d') + "&sig=qdjzni"
        page = urllib.urlopen(url)
        log = page.read()
        data = json.loads(log)
        user_list = []

        for item in data:
            if item['niversion'] == version:
                user_list.append(item['deviceid'])
        user_list = list(set(user_list))
        new_user = list(set(user_list).difference(set(pre_user))) # b中有而a中没有的
        pre_user.extend(new_user)
        new_user_num.append(len(new_user))
        sumlist.append(user_list)
        dates.append(date.strftime('%Y-%m-%d'))
    return sumlist,dates,new_user_num

def is_new_user_or_not(userlist,deviceid):
    if userlist.count(deviceid):
        return 0
    else:
        return 1

def compute_frequency_in_seven_days(deviceid, sumlist,i):

    sublist = list(flatten_list(sumlist[i:i+7]))
    frequency = sublist.count(deviceid)

    return frequency

def compute_percentage(frequency_list):
    user_sum_num = len(frequency_list)
    frequency_dict = {}
    for i in range(0,7):
        frequency_dict[i] = frequency_list.count(i)/float(user_sum_num)

    return frequency_dict


if len(argv)>3:
    duration = int(argv[2])
    version = argv[3]

if len(argv)>2:
    duration = int(argv[2])

start_date = argv[1]
start_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')
#process 17 days data
start_time = datetime.datetime.now()
user_list,dates,new_user_number = get_user_list_by_date(start_date)
end_time = datetime.datetime.now()
print(end_time-start_time)


frequency_list = []
frequency_sum_list = []
frequency_change = []
for j in range(0,duration):
    for i in range(j,j+7):
        for item in user_list[i]:
            frequency_list.append(compute_frequency_in_seven_days(item,user_list,i+1))

    dic = compute_percentage(frequency_list)
    frequency_sum_list.append(frequency_list)
    frequency_change.append(dic.values())

plt.figure(1)
for i in range(0,6):
    l = [x[i] for x in frequency_change]
    plotpic(dates[0:duration],l,"user use " + str(i) + " days in later seven days")

plt.legend(loc="upper right",prop={'size':6})

plt.figure(2)

plotpic(dates,new_user_number,"new user")

plt.legend(loc="upper right",prop={'size':6})
plt.show()