from django.shortcuts import render
from rest_framework.views import APIView
from .models import Medicine, Comment
#, MedicineElasticSearch
from .serializers import MedicineSerializer, MedicineDetailSerializer, CommentSerializer
#, MedicineElasticSearchSerializer, MedicineElasticSaveSerializer
from rest_framework.response import Response
from rest_framework.status import HTTP_204_NO_CONTENT, HTTP_400_BAD_REQUEST, HTTP_201_CREATED
from rest_framework.exceptions import NotFound, NotAuthenticated, PermissionDenied
from reviews.serializers import ReviewListSerializer
from .models import Medicine
from django.db.models import Q
from django.core.paginator import Paginator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import vision
import io
import os
import re
import time
from rest_framework.renderers import JSONRenderer
""" 엘라스틱 서치 모듈 """
from rest_framework.views import APIView
from rest_framework.response import Response
from elasticsearch import Elasticsearch

from .models import Medicine
from .serializers import MedicineSerializer


""" 이미지 서치 클래스 """
import torch.nn as nn
import torch.multiprocessing
import torch
from torchvision import transforms
from PIL import Image
import pickle

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = torch.nn.ReLU()
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=25088, out_features=94)

    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.layer3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = self.layer4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class image_matching:
    def __init__(self, model, img_path):
        self.model = model
        self.img_path = img_path

    def find(self):
        # 이미지 불러오기 및 전처리
        img = Image.open(self.img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1948, 0.2124, 0.2306], std=[0.0730, 0.0721, 0.0380])
        ])
        img_tensor = transform(img)

        # 모델 입력을 위해 차원 추가
        img_tensor = img_tensor.unsqueeze(0)  # batch size 1로 변경

        # 모델에 이미지 입력하여 예측 수행
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)

        # 예측 결과 출력
        _, predicted = torch.max(output.data, 1)

        predicted_num = predicted.item()
        return predicted_num

"""
class MedicineImageElasticSearch(APIView):
    def get(self, request, format=None):
            # 검색어 가져오기
            query = request.GET.get('q', '')
            start = time.time()
            # Elasticsearch 연결
            
            with open("D:/test_image/name_num_match.pickle", 'rb') as f:
                name_num_match = pickle.load(f)

            # 모델 정의와 모델 로드
            device = torch.device('cpu')
            model = CNN()
            model.load_state_dict(torch.load('D:/test_image/model.pt', map_location=device))
            test = image_matching(model, 'D:/test_image/test_image/2.png')
            medicine_name = name_num_match[test.find()]
            print(medicine_name)
            es = Elasticsearch()

            # 검색 쿼리 생성, 검색조건 수정가능한 부분.
            search_query = {
                "query": {
                    "multi_match": {
                        "query": medicine_name,
                        "fields": ["name^6", "basis", "effect", "caution", "cautionOtherMedicines"],
                        "type": "best_fields",
                        "operator": "and"
                    }
                }
            }

            # Elasticsearch 검색 수행
            search_results = es.search(index="medicine_index", body=search_query)

            # 검색 결과에서 Medicine 객체 가져오기
            medicine_ids = [hit['_id'] for hit in search_results['hits']['hits']]
            medicines = Medicine.objects.filter(id__in=medicine_ids)

            # Serializer로 json형태로 변환
            serializer = MedicineDetailSerializer(medicines, many=True)
            print("time :", time.time() - start)
            return Response(serializer.data)
"""

class MedicineImageSearch(APIView):
    def get(self, request):
        
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size
        start = time.time()

        with open("D:/test_image/name_num_match.pickle", 'rb') as f:
                name_num_match = pickle.load(f)

        device = torch.device('cpu')
        model = CNN()
        model.load_state_dict(torch.load('D:/test_image/model.pt', map_location=device))
        test = image_matching(model, 'D:/test_image/test_image/2.png')
        medicine_name = name_num_match[test.find()]

        print(medicine_name)
        multiSearch = medicine_name.split(",")

        q_object = Q()
        for t in multiSearch:
            print(t)
            q_object |= Q(name__icontains=t)

        pureresult = Medicine.objects.filter(q_object)
        print(pureresult)
        #result = pureresult[start:end]
        serializer = MedicineDetailSerializer(pureresult, many=True)
        print("time :", time.time() - start)
        return Response(serializer.data)

""" 엘라스틱서치 """
class MedicineSearchAPIView(APIView):
    def get(self, request, format=None):
        # 검색어 가져오기
        query = request.GET.get('q', '')
        start = time.time()
        # Elasticsearch 연결
        es = Elasticsearch()

        # 검색 쿼리 생성, 검색조건 수정가능한 부분.
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["name^6", "basis", "effect", "caution", "cautionOtherMedicines"],
                    "type": "best_fields",
                    "operator": "and"
                }
            }
        }

        # Elasticsearch 검색 수행
        search_results = es.search(index="medicine_index", body=search_query)

        # 검색 결과에서 Medicine 객체 가져오기
        medicine_ids = [hit['_id'] for hit in search_results['hits']['hits']]
        medicines = Medicine.objects.filter(id__in=medicine_ids)

        # Serializer로 json형태로 변환
        serializer = MedicineDetailSerializer(medicines, many=True)
        print("time :", time.time() - start)
        return Response(serializer.data)
    
""" 의약품 직접검색 """
class searchMedicineResult(APIView):
    def get(self, request):
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size
        start = time.time()
        search = request.GET.get('searchmedicine','')
        multiSearch = search.split(",")

        q_object = Q()
        for t in multiSearch:
            q_object |= Q(name__icontains=t)

        pureresult = Medicine.objects.filter(q_object)
        result = pureresult[start:end]
        serializer = MedicineDetailSerializer(pureresult, many=True)
        print("time :", time.time() - start)
        return Response(serializer.data)
    
""" 이미지 ocr 검색 """
class find_str:
  def __init__(self, json_path, image_path, df_str):
    self.json_path = json_path
    self.image_path = image_path
    self.df_str = df_str
    self.low_name = ['자모', '뇌선', '얄액', '쿨정']
    self.x_list = ['(', '[', '{', '<']
    self.remove_str = '_|"|'
    self.start_str = []
    self.end_str = []
  # 1. 불필요한 문자 찾기
  def num_stopword(self, DB_name_string):
    num_list = []
    # 불필요한 문자 위치 찾기
    for i in self.x_list:
      num = DB_name_string.find(i)
      # 없으면 pass
      if num == -1:
        pass
      else:
        num_list.append(num)
    # 제일 앞에 있는 특수문자 찾기
    num_list.sort()
    if len(num_list) != 0:
      str_stopword = DB_name_string[num_list[0]]
    # 특수문자가 없는 경우 DB에 없는 문자로 split 영향 없애기
    else:
      str_stopword = '?'
    return str_stopword
  def txt_extract(self):
    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.json_path
    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    a = []
    # The name of the image file to annotate
    file_name = os.path.abspath(self.image_path)
    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    # Performs text detection on the image file
    response = client.text_detection(image=image)
    texts = response.text_annotations
    texts_list = list(texts)
    txt_list = texts_list[0].description.split()
    return txt_list
  def searching(self):
    txt_list = self.txt_extract()
    df_str = self.df_str
    result = []
    trash_set = set(['조제약사', '에너지대사', '수납금액'])
    for txt in txt_list:
        str_stopword = self.num_stopword(txt)
    # 괄호 제거한 문자
        word = txt.split(str_stopword)[0]
        word = re.sub(self.remove_str, '', word)
        word = word.replace('*' ,'')
        if len(word) < 3 and txt not in self.low_name:
            pass
        else:
            for x, y in zip(df_str['start_str'], df_str['end_str']):
                if word[0] == x and word[-1] == y:
                    result.append(word)

    # 중복제거
    result = set(result)

    # trash 단어 제거
    result = result - trash_set
    return result


class searchOcrResult(APIView):
    def get(self, request):
        search = request.GET.get('searchocr','')
        json_path = "D:/test/ocrmedicine-86e789bdf085.json"
        image_path = "D:/test/IMG_4471.jpg"
        df_str = pd.read_csv('D:/db/df_str.csv')
        find = find_str(json_path, image_path, df_str)
        results = find.searching()
        results_list = list(results)
        print(results_list)
        
        q_object = Q()
        for t in results_list:
            q_object |= Q(name__startswith=t)

        result = Medicine.objects.filter(q_object)
        

        serializer = MedicineDetailSerializer(result, many=True)
        return Response(serializer.data)
    
    def SearchOCR(request):
        
        final_list = []
        content_list = Medicine.objects.all()

        json_path = "D:/test/ocrmedicine-86e789bdf085.json"
        image_path = "D:/test/IMG_4471.jpg"
        df_str = pd.read_csv('D:/db/df_str.csv')
        find = find_str(json_path, image_path, df_str)
        results = find.searching()
        results_list = list(results)
        print((results_list[1]))# set형식으로 여러개의 결과값 출력.
        
        # Query Set 
        #for result in results_list:
        if results_list[1]:
            ocr_result = content_list.filter(
            Q(name__icontains = results_list[1]),
            )
            #if ocr_result:
            #    final_list.append(ocr_result)
        
        print(ocr_result)
        serializer = MedicineDetailSerializer(ocr_result, many=True)
        return Response(serializer.data) 
        #return render(request, 'search_medicine.html',{'posts':posts, 'Boards':boards, 'result':result})

""" 증상검색 테스트 """
class SearchSymptom(APIView):
    def get(self, request):
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size

        content_list = Medicine.objects.all()
        search = request.GET.get('searchsymptom','')
        
        multiSearch = search.split(",")
        print(multiSearch)

        q_object = Q()
        for t in multiSearch:
            print(t)
            q_object |= Q(effect__icontains=t)

        pureresult = Medicine.objects.filter(q_object)

        result = pureresult[start:end]
        serializer = MedicineDetailSerializer(result, many=True)
        return Response(serializer.data) 


""" 기본 views """
class Medicines(APIView):

    def get(self, request):
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size    
        all_Medicines = Medicine.objects.all()[start:end]
        print(all_Medicines)
        serializer = MedicineSerializer(all_Medicines, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        if request.user.is_staff or request.user.is_superuser:
            serializer = MedicineSerializer(data=request.data)
            if request.user.is_authenticated:
                return NotAuthenticated
            if not request.user.is_staff or not request.user.is_superuser:
                return PermissionDenied
            if serializer.is_valid():
                new_medicine = serializer.save(permission_writer=request.user)
                return Response(MedicineSerializer(new_medicine).data)
            else:
                return Response(serializer.errors)
        else:
            raise NotAuthenticated


class MedicineDetail(APIView):

    def get_object(self, pk):
        try:
            return Medicine.objects.get(pk=pk)
        except Medicine.DoesNotExist:
            raise NotFound

    def get(self, request, pk):
        serializer = MedicineDetailSerializer(self.get_object(pk), context={'request':request},)
        return Response(serializer.data)
    
    def put(self, request, pk):
        medicine = self.get_object(pk)
        if not request.user.is_authenticated:
            raise NotAuthenticated
        if not request.user.is_staff or not request.user.is_superuser:
            raise PermissionDenied
        serializer = MedicineDetailSerializer(
            self.get_object(pk),
            data=request.data,
            partial=True,
            )
        if serializer.is_valid():
            updated_medicine = serializer.save(permission_writer=request.user)
            return Response(MedicineDetailSerializer(updated_medicine).data)
        else:
            return Response(serializer.errors)

    def delete(self, request, pk):
        medicine = self.get_object(pk)
        # 1. 유저가 아니면 삭제할 수 없다.
        if not request.user.is_authenticated:
            raise NotAuthenticated
        # 2. 작성자가 아니면 삭제할 수 없다.
        if not request.user.is_staff or not request.user.is_superuser:
            raise PermissionDenied
        medicine.delete()
        return Response(status=HTTP_204_NO_CONTENT)


class Comments(APIView):
    def get(self, request):
        all_Comments = Comment.objects.all()
        serializer = CommentSerializer(all_Comments, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = CommentSerializer(data=request.data)
        if serializer.is_valid():
            new_comment = serializer.save()
            return Response(CommentSerializer(new_comment).data)
        else:
            return Response(serializer.errors)
        # save하기 전에 serializer에서 user정보를 받아와야한다.


class MedicineReview(APIView):
    """ 리뷰에서 연동된 FK medicine을 활용하여 관련 리뷰 출력 """
    def get_object(self, pk):
        try:
            return Medicine.objects.get(pk=pk)
        except Medicine.DoesNotExist:
            raise NotFound
    
    def get(self, request, pk):
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size
        medicine = self.get_object(pk)
        serializer = ReviewListSerializer(
            medicine.reviews.all()[start:end],#[:]pagination! 엄청 심플하다. 사랑한다 장고
            many=True,
            )
        return Response(serializer.data)