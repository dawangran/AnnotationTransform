from setuptools import setup 
from setuptools import find_packages

setup(name='AnnotationTransform',
      version='0.0.1',
      description='Single cell automatic annotation',
      author='dawn',
      author_email='605547565@qq.com',
      requires= ['pandas','scanpy','numpy','sklearn','pickle'], # 定义依赖哪些模块
      packages=find_packages(),  # 系统自动从当前目录开始找包
      license="apache 3.0"
      )
