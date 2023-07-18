import requests
from django.http import JsonResponse
import json
import pandas as pd


def read_data_from_wechat(request):
    # 获取 access_token
    token_url = 'https://api.weixin.qq.com/cgi-bin/token'
     # 替换为你的 AppID
     # 替换为你的 AppSecret
    params = {
        'grant_type': 'client_credential',
        'appid': app_id,
        'secret': app_secret
    }
    response = requests.get(token_url, params=params)
    access_token = response.json().get('access_token')

    query_url = f'https://api.weixin.qq.com/tcb/databasequery?access_token={access_token}'
    payload = {
        'env': 'cloud1-1g8ztkqna96c8fc8',
        'query': 'db.collection("pred_history").get()'
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(query_url, json=payload, headers=headers)
    data = response.json()

    errcode = data.get('errcode')
    errmsg = data.get('errmsg')

    if errcode is not None and errcode != 0:
        # 处理错误，例如记录日志或返回错误消息给前端
        error_message = f"Error code: {errcode}, Error message: {errmsg}"
        # 返回错误消息的示例
        return JsonResponse({"error": error_message}, status=500)

    # 返回读取的数据



    # pager = data.get('pager')
    # offset = pager.get('Offset')
    # limit = pager.get('Limit')
    # total = pager.get('Total')
    data_list = data.get('data')

    # 遍历数据列表并处理每个元素
    processed_data = []
    for item in data_list:
        item_dict = json.loads(item)  # 将列表中的字符串解析为字典
        # item_id = item_dict['_id']
        openid = item_dict['_openid']
        stockID = item_dict['stockID']  
        closes = item_dict['closes']
        currentTime = item_dict['currentTime']
        tradeDates = item_dict['tradeDates']
        
        # 构建新的数据结构或字典，并添加到处理后的数据列表中
        processed_item = {
            'openid': openid,
            'stockID': stockID,
            'currentTime': currentTime,
            'tradeDates': tradeDates,
            'closes': closes
        }
        processed_data.append(processed_item)

    # 将处理后的数据转换为 DataFrame
    df = pd.DataFrame(processed_data)
    




    return JsonResponse({'data': df.to_dict('records')})
