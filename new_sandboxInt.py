from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import CodeInterpreter
l=CodeInterpreter.list()
for i in l:
    print(i,"\n\n")
    
    CodeInterpreter.kill(i.sandbox_id)
e=CodeInterpreter()
e.keep_alive(3600)
print("is_open",e.is_open)
print('id: \n',e.id)