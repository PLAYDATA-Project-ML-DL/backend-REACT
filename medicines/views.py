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


""" 엘라스틱 서치 테스트 """
from rest_framework.views import APIView
from rest_framework.response import Response
from elasticsearch import Elasticsearch

from .models import Medicine
from .serializers import MedicineSerializer


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