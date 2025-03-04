from make_pdf import make_pdf

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import configparser
import openai
import pickle
import uvicorn
import os


app = FastAPI()    


# 모델 - GPT 3.5 Turbo 선택
model = os.getenv("MODEL")
openai.api_key = os.getenv("OPENAI_API_KEY")



# ChatGPT
def chatgpt_reply(message, title, path="./report.txt"):

    messages = []
    messages.append({"role": "system", "content": "당신은 낙상사고에 관한 전문가입니다."})
    messages.append({"role": "user", "content": f"{message}"})

    response = openai.ChatCompletion.create(
            model=model,
            messages=messages
    )

    assistant_content = response.choices[0].message["content"].strip()

    # 파일 열기
    with open(path, 'a', encoding='utf-8') as file:
        title_format = "[" + title + "]"
        file.write(title_format + "\n")
        file.write(assistant_content + "\n\n")


    return assistant_content


def makeDiseasePrompt(name, servey_result):

    prom = "다음은 " + name + "님의 질병 유무 조사 결과를 나타낸 것입니다. " + name + "님은 현재 "

    dis_count = 0
    if servey_result[5]['answer'] == "예":
        prom = prom + "고혈압, "
        dis_count = dis_count + 1
    if servey_result[6]['answer'] == "예":
        prom = prom + "당뇨, "
        dis_count = dis_count + 1
    if servey_result[7]['answer'] == "예":
        prom = prom + "관절염, "
        dis_count = dis_count + 1
    if servey_result[8]['answer'] == "예":
        prom = prom + "심장질환, "
        dis_count = dis_count + 1
    if servey_result[9]['answer'] == "예":
        prom = prom + "뇌질환, "
        dis_count = dis_count + 1
    if servey_result[10]['answer'] == "예":
        prom = prom + "골다공증, "
        dis_count = dis_count + 1
    if servey_result[11]['answer'] == "예":
        prom = prom + "기립성 저혈압, "
        dis_count = dis_count + 1
    if servey_result[12]['answer'] == "예":
        prom = prom + "파킨슨병, "
        dis_count = dis_count + 1
    if servey_result[13]['answer'] == "예":
        prom = prom + "백내장, "
        dis_count = dis_count + 1
    if servey_result[14]['answer'] == "예":
        prom = prom + "녹내장, "
        dis_count = dis_count + 1
    if servey_result[15]['answer'] == "예":
        prom = prom + "황변변성, "
        dis_count = dis_count + 1
    if servey_result[16]['answer'] == "예":
        prom = prom + "빈뇨, "
        dis_count = dis_count + 1
    if servey_result[17]['answer'] == "예":
        prom = prom + "긴박뇨, "
        dis_count = dis_count + 1
    if servey_result[18]['answer'] == "예":
        prom = prom + "자다가 일어나서 화장실에 자주가는 증상, "
        dis_count = dis_count + 1
    if servey_result[19]['answer'] == "예":
        prom = prom + "설사 혹은 변비, "
        dis_count = dis_count + 1
    if servey_result[20]['answer'] == "예":
        prom = prom + "요실금 혹은 잔변, "
        dis_count = dis_count + 1


    if dis_count != 0:
        prom = prom + "과 같은 " + str(dis_count) + "개의 질병을 앓고 계십니다." + \
        " 각각의 질병들이 낙상 위험도를 높이는 이유를 설명하고 이를 항목별로 나누어 설명해주세요."
    else:
        prom = "특별한 질병을 가지고 계시지 않은 것으로 보입니다. '낙상 위험도를 높히는 질병을 가지고 계시지 않는 것으로 보입니다. 지금의 건강상태를 유지하신다면 일상생활에서의 낙상 위험도가 높아지지 않으실 것으로 보입니다.'라는 문장을 답변으로 출력하세요."


    return dis_count, prom




def makeBodyPrompt(name, servey_result, product):

    prom = "다음은 " + name + "님의 신체정보에 대해 조사한 결과를 나타낸 것입니다. " + name + "님은 현재 "

    body_count = 0
    if servey_result[21]['answer'] == "예":
        prom = prom + "가끔 보행시 움직임이 둔해진걸 느끼는 증상(운동프로그램 추천), "
        body_count = body_count + 1
    if servey_result[22]['answer'] == "예":
        prom = prom + "집 안에서 걸어다닐 때, 가구를 잡아야 한다든지 몸을 기대어 의지해야만 몸을 지탱할 수 있는 증상(기립보조손잡이, 천장손잡이, 부착형 안전손잡이 추천), "
        body_count = body_count + 1
        product.append((1, "BATHROOM"))
        product.append((9, "ROOM"))
    if servey_result[23]['answer'] == "예":
        prom = prom + "앉았다 일어설 때 의자나 방바닥을 짚고 손에 힘을 주어야만 하는 증상(기립보조손잡이 추천), "
        body_count = body_count + 1
        product.append((1, "BATHROOM"))
        product.append((9, "ROOM"))
    if servey_result[24]['answer'] == "예":
        prom = prom + "모퉁이를 돌거나 걷는 방향을 바꾸려고 할 때의 걸음걸이가 쉽지 않은 증상(지팡이 보행보조기 추천), "
        body_count = body_count + 1
        product.append((12, "ENTRANCE"))
    if servey_result[25]['answer'] == "예":
        prom = prom + "지팡이나 실버카 등 보행기를 사용해야만 이동할 수 있는 증상(지팡이 보행보조기 추천), "
        body_count = body_count + 1
        product.append((12, "ENTRANCE"))


    if body_count != 0:
        prom = prom + "과 같은 " + str(body_count) + "개의 낙상 위험도를 높이는 신체 증상을 가지고 계십니다." + \
        " 낙상 위험도를 낮추기 위해서 어떤 해결책을 제시해주면 좋을까요? 괄호 안에 내용은 각 증상에 대한 해결책입니다. 각각의 해결책을 솔루션으로 제시하시고 그 이유를 문단으로 정리해서 증상별로 1,2,3... 순서로 정리해서 진단 결과를 말해주세요."
    else:
        prom = "신체에 문제가 있어보이진 않아보입니다. '낙상 위험도를 높히는 질병을 가지고 계시지 않는 것으로 보입니다. 지금의 건강상태를 유지하신다면 일상생활에서의 낙상 위험도가 높아지지 않으실 것으로 보입니다.'라는 문장을 답변으로 출력하세요."



    return body_count, prom



def makeHouseSolution(servey_result, product, index_10):

    # 어떤 물건을 추천해야하는지 확인
    product_vector = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    recommendation_comment = [
        "집안에 10센티 이상의 계단, 문턱, 또는 단차가 있다면, 높은 문턱에 걸려 넘어져 낙상 사고가 발생할 수 있습니다. 낙상 위험을 예방하기 위해 '문턱단차해소기'를 추천합니다. 이 제품은 문턱과 단차에 걸려넘어지는 일을 예방하고 노인분들의 안전한 이동 환경 조성에 도움을 주어 노인 및 기타 취약 계층의 낙상 사고를 예방할 수 있습니다.",
        "집안 내부 문턱이나 계단 주변에 안전손잡이가 설치되어 있지 않다면, 문턱을 넘거나 계단을 오르내리는 과정에서 미끄러져 넘어지는 사고가 발생할 수 있습니다. 낙상 예방을 위해 '안전손잡이'를 설치하는 것을 강력히 권장합니다. 특히 노인 분 혹은 이동에 어려움을 겪는 분들에게 손잡이는 계단 오르내림이나 문턱을 넘어갈 때 신체의 균형을 유지하며 도움을 주어, 낙상 위험을 줄일 수 있습니다.",
        "화장실 입구에 10cm이상의 문턱이나 단차가 있으시다면, 높은 문턱에 걸려 넘어져 낙상 사고가 발생할 수 있습니다. 낙상 예방을 위해 '안전손잡이'를 설치하는 것을 강력히 권장합니다. 화장실 출입시 문턱을 넘어갈 때 신체의 균형을 유지하는데 도움을 주어, 낙상 위험을 줄일 수 있습니다.",
        "화장실 들어갈 때 30cm이상의 문턱이나 단차가 있어서 넘어가기 어려우신 경우, 높은 문턱에 걸려 넘어져 낙상 사고가 발생할 수 있습니다. 낙상 예방을 위해 '논슬립디딤발판'을 권장합니다. 높은 문턱과 단차를 넘어갈 때 논슬립발판을 밟고 넘어감으로써 단차의 높이를 간접적으로 낮추고 단차로 인해 넘어져 겪는 사고를 방지합니다.",
        "욕실 바닥의 타일 재질이 미끄러우시다면 화장실내 이동중에 미끄러운 바닥으로 인해 넘어질 가능성이 크게 증가합니다. 이러한 사고를 방지하기 위해 '안전손잡이', '퍼즐매트', 그리고 '논슬립 테이프 바닥시트'를 추천드립니다. 안전손잡이는 미끄러운 바닥을 걸을 때 손잡이를 잡고 몸을 지지하는데 도움을 줍니다. 그리고 퍼즐매트와 논슬립 테이프 바닥시트는 바닥의 미끄러운 정도를 줄여줌으로써 화장실에서 미끄러지고 넘어지는 사고를 방지합니다.",
        "변기에 앉고 설 때 지지대가 없으시다면 '변기거치형 안전손잡이'를 추천드립니다. 변기에 앉고 일어서실 때, 손잡이를 잡고 움직임으로써 몸의 균형을 유지하고 안정성을 높혀 화장실에서 미끄러지고 넘어지는 낙상 사고 위험을 낮출 수 있습니다.",
        "욕조 주변이나 샤워하시는 곳 벽면에 안전손잡이가 없으시다면 물로 인해 미끄러워진 바닥 때문에 넘어지는 사고가 발생할 가능성이 높습니다. 이를 방지하고자 '안전손잡이'를 추천드립니다. 욕조 출입시에 안전손잡이를 잡고 이동하심으로써 미끄러운 바닥으로 인해 넘어지는 사고를 방지할 수 있으며 낙상으로 인한 사고를 미리 예방할 수 있습니다.",
        "욕실에 앉아 목욕을 하거나, 손빨래 등 물을 사용하는 경우가 많으시다면 이동 중에 물로 인해 미끄러워진 바닥에서 넘어질 가능성이 높습니다. 낙상 사고를 예방하기위해 '목욕의자', '욕실매트'를 추천드립니다. 목욕의자는 화장실에서 앉아있는 동안 자세의 안정성을 유지시킬 수 있으며 욕실매트는 바닥의 미끄러운 정도를 낮춤으로써 낙상 사고 위험을 예방할 수 있습니다.",
        "침실이나 거실 바닥이 미끄러우시다면 이동 중 미끄러운 바닥으로 인해 넘어질 가능성을 높힙니다. 이를 예방하기 위해서 '접이식 미끄럼 방지매트', '미끄럼방지매트' 그리고 '안전손잡이'를 추천드립니다. 미끄럼 방지매트류는 침실이나 거실 바닥의 미끄러운 정도를 줄이는데 도움이 되며 안전손잡이는 거실과 침실에서 이동시에 몸의 균형을 안정적으로 유지하는데 도움을 줄 수 있을 것입니다.",
        "침실 전등을 끄는 스위치가 침대에서 멀리위치하신다면 소등을 위해 이동하는 과정에서 넘어지는 사고가 발생할 수 있습니다. 낙상 사고를 예방하기 위해 '점소등 리모컨' 그리고 '침대난간 거치형 안전손잡이'를 추천드립니다. 점소등 리모컨은 원격으로 스위치를 조절함으로써 소등시 이동 과정에서 발생하는 낙상 사고를 예방할 수 있습니다. 그리고 침대난간 거치형 안전손잡이는 침대에서 기상시 몸을 지탱하는데 도움을 줌으로써 넘어지는 사고를 방지합니다.",
        "거실이나 침실 문턱이나 단차가 있어서 걸려 넘어진 적이 있다면 '문턱단차해소기'와 '논슬립 디딤발판'을 추천드립니다. 문척단차해소기는 높은 단차와 문턱을 넘나들 때의 높이를 낮추는데 도움을 드릴 수 있으며 디딤발판을 단차나 문턱 사이를 더 적은 힘과 안전하게 이동할 수 있게 도움을 드릴 것입니다.",
        "집안내부 가구배치나 통로의 폭이 이동하기에 걸리고, 좁은 편이시라면 이동 중 장애물에 걸려 넘어질 가능성이 높습니다. 낙상 사고를 방지하기 위해 '가구의 이동과 집청소정리 지원 서비스'를 추천드립니다. 좁은 통로는 작은 장애물에도 더 쉽게 넘어질 수 있도록 하며 더 큰 사고를 발생시킬 수 있습니다. 이를 위해 가구를 이동시키고 집을 청소함으로써 이동시에 더 넓은 공간을 확보하시길 바랍니다.",
        "주방 부엌의 바닥이 물기 등으로 인해 미끄러운 편이시라면 미끄러운 바닥으로 인해 넘어질 가능성이 높아집니다. 낙상 사고를 사전에 예방하기 위해 '미끄럼 방지 매트와 접이식 미끄럼 방지매트'를 추천드립니다. 물기로 인해 미끄러운 바닥을 미끄럼 방지 매트가 미끄러운 정도를 낮춰드리며 결과적으로 낙상으로 인한 사고를 예방하는데 큰 도움이 될 것입니다."
    ]



    # 현관 입구/출입구
    if servey_result[index_10+2]['answer'] == "예": # 12-1
        product.append((11, "ENTRANCE"))
        product_vector[0] = 1
    if servey_result[index_10+3]['answer'] == "아니요": # 12-2
        product.append((1, "ENTRANCE"))
        product_vector[1] = 1


    # 욕실
    if servey_result[index_10+4]['answer'] == "예": # 13-1
        product.append((1, "BATHROOM"))
        product_vector[2] = 1
    if servey_result[index_10+5]['answer'] == "예": # 13-2
        product.append((2, "BATHROOM"))
        product_vector[3] = 1
    if servey_result[index_10+6]['answer'] == "예": # 13-3
        product.append((1, "BATHROOM"))
        product.append((3, "BATHROOM"))
        product.append((6, "BATHROOM"))
        product_vector[4] = 1
    if servey_result[index_10+7]['answer'] == "아니요": # 13-4
        product.append((5, "BATHROOM"))
        product_vector[5] = 1
    if servey_result[index_10+8]['answer'] == "아니요": # 13-5
        product.append((1, "BATHROOM"))
        product_vector[6] = 1
    if servey_result[index_10+9]['answer'] == "예": # 13-6
        product.append((3, "BATHROOM"))
        product.append((4, "BATHROOM"))
        product_vector[7] = 1


    # 침실/거실
    if servey_result[index_10+10]['answer'] == "예": # 14-1
        product.append((8, "ROOM"))
        product.append((1, "ROOM"))
        product_vector[8] = 1
    if servey_result[index_10+11]['answer'] == "아니요": # 14-2
        product.append((10, "ROOM"))
        product.append((9, "ROOM"))
        product_vector[9] = 1
    if servey_result[index_10+12]['answer'] == "예": # 14-3
        product.append((11, "ROOM"))
        product.append((2, "ROOM"))
        product_vector[10] = 1
    if servey_result[index_10+13]['answer'] == "아니요": # 14-4
        product_vector[11] = 1
    if servey_result[index_10+14]['answer'] == "예": # 14-5
        product.append((8, "KITCHEN"))
        product_vector[12] = 1

    return product_vector, recommendation_comment



# 낙상 위험 수준 예측
def predict_level(servey_result, index_10):
    target_vector = [] # 등급을 predict할 벡터 (설문지가 달라지면서 문항 번호도 바뀌었음. 이를 반영하여 입력)
    target_vector.append(servey_result[index_10+15]['weight']) # 기존 1번 (성별)
    target_vector.append(servey_result[index_10+16]['weight']) # 기존 2번 (연도생)
    target_vector.append(servey_result[0]['weight']) # 기존 3번 (현재 1번)
    target_vector.append(servey_result[index_10]['weight']) # 기존 4번 (현재 10번)
    target_vector.append(servey_result[index_10+1]['weight']) # 기존 6번 (현재 11번)
    target_vector.append(servey_result[1]['weight']) # 기존 9번 (현재 2번)
    target_vector.append(servey_result[21]['weight']) # 기존 10-1번 (현재 7-1번)
    target_vector.append(servey_result[22]['weight']) # 기존 10-2번 (현재 7-2번)
    target_vector.append(servey_result[23]['weight']) # 기존 10-3번 (현재 7-3번)
    target_vector.append(servey_result[24]['weight']) # 기존 10-4번 (현재 7-4번)
    target_vector.append(servey_result[25]['weight']) # 기존 10-5번 (현재 7-5번)
    target_vector.append(servey_result[13]['weight']) # 기존 11-1번 (현재 5-1번)
    target_vector.append(servey_result[14]['weight']) # 기존 11-2번 (현재 5-2번)
    target_vector.append(servey_result[15]['weight']) # 기존 11-3번 (현재 5-3번)
    target_vector.append(servey_result[5]['weight']) # 기존 12-1번 (현재 4-1번)
    target_vector.append(servey_result[6]['weight']) # 기존 12-2번 (현재 4-2번)
    target_vector.append(servey_result[7]['weight']) # 기존 12-3번 (현재 4-3번)
    target_vector.append(servey_result[8]['weight']) # 기존 12-4번 (현재 4-4번)
    target_vector.append(servey_result[9]['weight']) # 기존 12-5번 (현재 4-5번)
    target_vector.append(servey_result[10]['weight']) # 기존 12-7번 (현재 4-6번)
    target_vector.append(servey_result[11]['weight']) # 기존 12-8번 (현재 4-7번)
    target_vector.append(servey_result[12]['weight']) # 기존 12-11번 (현재 4-8번)
    target_vector.append(servey_result[16]['weight']) # 기존 13-1번 (현재 6-1번)
    target_vector.append(servey_result[17]['weight']) # 기존 13-2번 (현재 6-2번)
    target_vector.append(servey_result[18]['weight']) # 기존 13-3번 (현재 6-3번)
    target_vector.append(servey_result[19]['weight']) # 기존 13-4번 (현재 6-4번)
    target_vector.append(servey_result[20]['weight']) # 기존 13-5번 (현재 6-5번)
    target_vector.append(servey_result[26]['weight']) # 기존 16번 (현재 8번)
    target_vector.append(servey_result[27]['weight']) # 기존 27번 (현재 9-1번)

    target_vector = np.array(target_vector)
    target_vector = target_vector.reshape(1, -1)

    # Load the model
    with open('./kmeans_best_model', 'rb') as f:
        loaded_kmeans_model = pickle.load(f)
        cluster_label = loaded_kmeans_model.predict(target_vector)


    return cluster_label



@app.post("/makeReport")
async def generate_report(servey : dict):

    name = servey['name']
    data = servey['data']
    goods_list = []

    # 10번 질문이 시작하는 시점
    if len(data) > 53:
        start10 = 37
    else:
        start10 = 28


    ''' goods_list

        1. 안전손잡이
        2. 논슬립 디딤발판
        3. 퍼즐매트
        4. 목욕의자
        5. 변기거치형 안전손잡이
        6. 논슬립테이프 바닥시트
        7. 접이식 미끄럼방지매트 > 제외
        8. 미끄럼방지매트
        9. 침대난간거치형 안전손잡이
        10. 점소등 스위치
        11. 문턱단차해소기
        12. 스마트지팡이
        13. 벽걸이 안전 의자
        14. 가스 안전 타이머
        15. 간이 변기
        16. 논슬립테이프
    
    '''

    # 설정파일 읽기
    config = configparser.ConfigParser()    
    config.read('./config.ini', encoding='utf-8')
    folder_path = config['makeReport']['txt_file_path']
    folder_path_pdf = config['makeReport']['pdf_folder_path']


    # 현재 날짜 받아오기
    current_time = datetime.now()
    # 날짜, 시간 포맷 설정
    current_time_string = current_time.strftime("%y%m%d%H%M") 

    # 새로운 파일 이름 생성
    new_file_name = f"{name}_{current_time_string}.txt"
    new_file_path = os.path.join(folder_path, new_file_name)

    # 기본정보 - 1번, 2번 질문
    base_prompt = name + "님은 " + data[start10+15]['answer'] + "이며 " + data[start10+16]['answer'] + '생입니다.'
    base_prompt = base_prompt + "현재 2023년도를 기준으로 나이를 계산하여 결과를 출력할때는 \"OOO씨 / 성별 / OO년생(나이)\" 형태로만 대답해주세요."
    chatgpt_reply(base_prompt, "기본 신체 정보", new_file_path)

    # 질병정보
    counting, disease_prompt = makeDiseasePrompt(name, data)

    if counting != 0:
        chatgpt_reply(disease_prompt, "질환 정보", new_file_path)
    else: # 학습 안됨 > 질환 정보 0개면 이렇게 표기
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "질환 정보" + "]"
            file.write(temp_title_format + "\n")
            file.write(name + "님은 낙상 위험도를 높히는 질병을 가지고 있지 않은 것으로 보입니다. 현재의 건강 상태를 유지하신다면 낙상 위험도를 계속 낮추실 수 있을 것으로 보입니다." + "\n\n")


    # 신체정보
    counting, body_prompt = makeBodyPrompt(name, data, goods_list)

    if counting != 0:
        chatgpt_reply(body_prompt, "신체 정보", new_file_path)
    else:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "신체 정보" + "]"
            file.write(temp_title_format + "\n")
            file.write(name + "님은 낙상 위험도를 신체 상태을 가지고 있지 않은 것으로 보입니다. 현재의 신체 건강을 유지하신다면 낙상 위험도를 계속 낮추실 수 있을 것으로 보입니다." + "\n\n")


    # 운동주기
    sports = str(data[26])
    sports_prompt = "다음 질문과 응답은 " + name + "님의 운동주기에 관한 설문조사 결과입니다." + sports + \
                    name + "님은 운동 주기를 낙상 사고 위험도와 연관지어 판단하여 진단한 내용을 요약하여 보여주세요. 운동 주기가 주 3회보다 적으면 운동을 많이하도록 권장하고 왜 운동을 자주하는 것이 낙상위험도를 낮출 수 있는지도 함께 설명해주세요. 그리고 주 3회보다 많이하는 사람이라면 계속해서 지금과 같은 운동 습관을 유지하도록 권장하세요."
    chatgpt_reply(sports_prompt, "운동 주기", new_file_path)

    # 낙상 경험
    fall_experience = str(data[27])
    fall_experience_prompt = "다음 질문과 응답은 " + name + "님의 낙상사고 경험에 관하여 조사한 설문조사 결과입니다." + fall_experience + \
                            "이런 정보들을 종합하여 " + name + "님의 낙상 사고 위험도를 판단하고 그 이유가 무엇인지 서술해주세요. 낙상 경험이 1회 이상인 경우, 일반 사람들에비해 낙상 위험도가 높은 편이니 이를 기준으로 판단해주세요."
    chatgpt_reply(fall_experience_prompt, "낙상 경험", new_file_path)


    # 기본 주거정보 - 3, 4번 항목
    base_live_info = str(data[0]) + str(data[start10])
    base_live_prompt = "다음 질문과 응답은 " + name + "님의 주거 정보와 같이 살고 있는 사람의 유무를 조사한 설문조사 결과입니다." + base_live_info + \
                        "이런 주거에 관한 기본정보를 한 문장으로 출력해주세요."
    chatgpt_reply(base_live_prompt, "기본 주거 정보", new_file_path)


    # 주거 낙상 솔루션
    recommend_list, comment = makeHouseSolution(data, goods_list, start10)
    weight_vector = [7, 6, 6, 8, 9, 6, 8, 6, 5, 6, 4, 4, 6]


    if np.sum(recommend_list) == 0: # 추천할 제품 없음
        top_comments = []
    else: # 물건을 하나라도 추천함.
        rec = np.array(recommend_list)
        wei = np.array(weight_vector)
        mul_result = rec * wei
        nonzero_indices = np.nonzero(mul_result)[0]

        top_indices = np.argsort(mul_result[nonzero_indices])[-3:][::-1]
        other_indices = []
        top_comments = [comment[i] for i in nonzero_indices[top_indices]]

        for index, _ in enumerate(nonzero_indices):
            if index not in top_indices:
                other_indices.append(index)



    # 주거환경진단 1 (현관/입구)
    if recommend_list[0] == 1 or recommend_list[1] == 1:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "주거환경진단 1: 현관/입구" + "]"
            file.write(temp_title_format + "\n")
            for i in range(2):
                if recommend_list[i] == 1:
                        file.write(comment[i] + "\n")
    else:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "주거환경진단 1: 현관/입구" + "]"
            file.write(temp_title_format + "\n")
            file.write("현관 입구에서의 낙상 위험 가능성은 높지 않은 것으로 진단되었습니다.\n")


    # 주거환경진단 2 (화장실/욕실)
    if recommend_list[2] == 1 or recommend_list[3] == 1 or recommend_list[4] == 1 or recommend_list[5] == 1 or recommend_list[6] == 1 or recommend_list[7] == 1:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "주거환경진단 2: 화장실/욕실" + "]"
            file.write(temp_title_format + "\n")
            for i in range(2, 8):
                if recommend_list[i] == 1:
                        file.write(comment[i] + "\n")
    else:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "주거환경진단 2: 화장실/욕실" + "]"
            file.write(temp_title_format + "\n")
            file.write("화장실/욕실에서의 낙상 위험 가능성은 높지 않은 것으로 진단되었습니다.\n")
    

    # 주거환경진단 3 (거실/침실)
    if recommend_list[8] == 1 or recommend_list[9] == 1 or recommend_list[10] == 1 or recommend_list[11] == 1:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "주거환경진단 3: 거실/침실" + "]"
            file.write(temp_title_format + "\n")
            for i in range(8, 12):
                if recommend_list[i] == 1:
                        file.write(comment[i] + "\n")
    else:
        with open(new_file_path, 'a', encoding='utf-8') as file:
            temp_title_format = "[" + "주거환경진단 3: 거실/침실" + "]"
            file.write(temp_title_format + "\n")
            file.write("거실/침실에서의 낙상 위험 가능성은 높지 않은 것으로 진단되었습니다.\n")

    # 주거환경진단 4 (주방)
    with open(new_file_path, 'a', encoding='utf-8') as file:
        temp_title_format = "[" + "주거환경진단 4: 주방" + "]"
        file.write(temp_title_format + "\n")

        if recommend_list[12] == 1:
            file.write(comment[12] + "\n")
        else:
            file.write("주방에서의 낙상 위험 가능성은 높지 않은 것으로 진단되었습니다.\n")


    # 등급 예측
    predict_label = predict_level(data, start10)
    final_level = int(predict_label[0])

    if final_level == 1: # 최종 낙상 위험 등급 판단
        final_level = 1
    elif final_level == 0:
        final_level = 2
    else:
        final_level = 3


    # 종합평가
    if len(top_comments) > 0:
        total_prompt = "낙상 위험등급은 1,2,3 등급으로 나뉘며 1등급은 매우 위험, 2등급은 보통 그리고 3등급은 낮음 입니다." + name + f"님은 {final_level}입니다. 이를 통해 낙상 위험 수준과 " + " 이 정보와 함께 다음에 제시된 낙상 예방을 위한 솔루션들을 함께 요약하여 200자내로 하나의 문단으로 작성해주세요. 200자를 절대 넘지 말아주세요." + str(top_comments)
    else:
        total_prompt = "낙상 위험등급은 1,2,3 등급으로 나뉘며 1등급은 매우 위험, 2등급은 보통 그리고 3등급은 낮음 입니다." + name + f"님은 {final_level}입니다. 이를 통해 낙상 위험 수준을 평가하여 요약해서 200자내로 하나의 문단으로 작성해주세요. 200자를 절대 넘지 말아주세요."
    
    summary = chatgpt_reply(total_prompt, "종합 평가", new_file_path)

    ############################# pdf 파일 저장 #############################
    # pdf 저장 폴더가 존재하지 않을 경우 폴더를 생성합니다.
    if not os.path.exists(folder_path_pdf):
        os.makedirs(folder_path_pdf)

    ######################## 절대 경로로 변경 필요 ########################
    # 새로운 docx 파일 이름 생성
    # docx, pdf파일 저장 위치
    new_docx_name = f"{current_time_string}_{name}.docx" # docx 파일 이름
    new_pdf_name = f"{current_time_string}_{name}.pdf" # pdf 파일 이름
    new_docx_path = os.path.join(folder_path_pdf, new_docx_name) # folder_path_pdf는 pdf를 저장하는 폴더 위치
    new_pdf_path = os.path.join(folder_path_pdf, new_pdf_name) # folder_path_pdf만 바꾸면 저장위치 자유롭게 바뀜


    # PDF 만들기
    if len(top_comments) > 0:
        make_pdf(name, current_time, nonzero_indices[top_indices], new_file_path, new_docx_path, folder_path_pdf)
    else:
        make_pdf(name, current_time, [-1], new_file_path, new_docx_path, folder_path_pdf)
        
    os.remove(new_file_path) # 생성했던 txt 파일 삭제
    os.remove(new_docx_path) # 생성했던 docx 삭제 - 리눅스만 가능


    goods_set = set(goods_list)
    goods_list = list(goods_set)

    goods_result = []
    for item in goods_list:
        item_id, item_location = item
        goods_result.append({"id": item_id, "location": item_location})

    # 이름_날짜
    url = new_pdf_path
    return {'rank': final_level, 'report': url, 'summary': summary, 'product': goods_result}


if __name__ == "__main__" :
    uvicorn.run(app, host="localhost", port=os.getenv("PORT", 8080))