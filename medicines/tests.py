#from django.test import TestCase
from rest_framework.test import APITestCase
# rest framework의 테스트가 훨씬 도움이 된다.
from . import models

class TestMedicines(APITestCase):
    # get, post 테스트
    NAME = 'Medicine name'
    BASIS = 'Medicine basis'

    def setUp(self):
        """ 데이터 베이스를 설정할 수 있는 함수 """
        models.Medicine.objects.create(
            name=self.NAME,
            basis=self.BASIS,
        )

    def test_two_plus_two(self):

        self.assertEqual(2+2, 4, "The math is wrong.") # param1, param2가 같지 않으면 false, param3(msg)가 출력된다.
        # 같은지 확인하는 테스트 함수

    def test_all_medicines(self):
        response = self.client.get("/api/v1/medicines/")
        # clinet : API client로 API서버로 request를 보낼 수 있게 해준다.
        # url을 쓰면 브라우저로 접근하는것 처럼 request를 전송한다.
        data = response.json()
        # 페이지에서 출력하는 json코드를 받는다, 공개된 페이지여야 하는 조건이 있다.
        # 데이터는 실제 DB에 있고, 테스트시 비어있는 임시 DB를 생성후 제거하기 때문에 실물 데이터는 볼 수 없다
        # 하지만 반환받는 데이터 타입을 통해 결과를 확인할 수 있다.
        
        self.assertEqual(response.status_code, 200, "Status code isn't 200.")
        # response, 접근 코드가 200(성공)인지 테스트
        self.assertIsInstance(data, list)
        # data의 type이 list인지 테스트
        self.assertEqual(len(data), 1,)
        # data의 길이가 1인지 테스트
        self.assertEqual(data[0]["name"], self.NAME,)
        # data의 0번째 요소의 name의 값이 self.NAME과 같은지 테스트