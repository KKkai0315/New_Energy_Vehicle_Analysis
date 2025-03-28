import json
import os
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest



# 创建AcsClient实例
client = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)
request = CommonRequest()

# domain和version是固定值
request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
request.set_version('2020-06-29')

# action name可以在API文档里查到
request.set_action_name('GetWsChGeneral')

# 需要add哪些param可以在API文档里查到
request.add_query_param('ServiceCode', 'alinlp')
request.add_query_param('Text', '市区基本不用油  平均三个月加一次油  高速油耗令人满意。速度快，城市道路不怕谁  去加油站都不知道要加多少号的油了 哈哈')
request.add_query_param('TokenizerId', 'GENERAL_CHN')
request.add_query_param('OutType', '0')

response = client.do_action_with_exception(request)
resp_obj = json.loads(response)
print(resp_obj)



request = CommonRequest()

# domain和version是固定值
request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
request.set_version('2020-06-29')

# action name可以在API文档里查到
request.set_action_name('GetSaChGeneral')

# 需要add哪些param可以在API文档里查到
request.add_query_param('ServiceCode', 'alinlp')
request.add_query_param('Text', '市区基本不用油  平均三个月加一次油  高速油耗令人满意。速度快，城市道路不怕谁  去加油站都不知道要加多少号的油了 哈哈')
response = client.do_action_with_exception(request)
resp_obj = json.loads(response)
print(resp_obj)


# domain和version是固定值
request.set_domain('alinlp.cn-hangzhou.aliyuncs.com')
request.set_version('2020-06-29')

# action name可以在API文档里查到
request.set_action_name('GetPosChGeneral')

# 需要add哪些param可以在API文档里查到
request.add_query_param('ServiceCode', 'alinlp')
request.add_query_param('Text', '周边口碑气温前方营造帮助衰减优化利用独立三电手套长度软性上牌便利产品好评大面积改变全速方向小巧频繁提醒更喜欢全新地板出远门计算积极家族工艺维修缺乏现代通风碰撞平顺性跑车')
response = client.do_action_with_exception(request)
resp_obj = json.loads(response)
print(resp_obj)
