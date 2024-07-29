# from logging_module import log_it
# import utils


# class :
#     def __getattribute__(self, name):
#         # Intercept all attribute access
#         attr = object.__getattribute__(self, name)
        
#         # Check if the attribute is a callable method
#         if callable(attr):
#             def new_attr(*args, **kwargs):
#                 # Call the specific function before the original method
#                 if name=="__call__":
#                     self.before_method_call(name, *args, **kwargs)
                
#                 # Call the original method
#                 result = attr(*args, **kwargs)
                
#                 return result
            
#             return new_attr
#         else:
#             return attr
#     def before_method_call(self, method_name, *args, **kwargs):
#         # Define the function to be called before the method
#         print(args)
#         class_name=type(self).__name__
#         log_it()
#         print(f"Calling method: {method_name} with args: {args} and kwargs: {kwargs}")
    
from langchain_core.callbacks import BaseCallbackHandler
# from langchain_core.callbacks.manager import 

class MyCustomHandler(BaseCallbackHandler):
    def on_custom_event(name: str, data: Any, *, run_id: UUID, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any):
        print(name data: Any, *, run_id: UUID, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any))