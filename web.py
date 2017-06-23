#coding=utf-8
import tornado.ioloop
import tornado.web
from predict import Predict
from tornado.escape import json_decode
import json

global model
model = Predict()

class MainHandler(tornado.web.RequestHandler):


    def post(self,*args,**kwargs):
        auth = self.get_body_argument('auth')
        uid = self.get_body_argument('uid')
        ques = self.get_body_argument('ques').encode('utf-8')

        if auth not in ['cloud']:
            return 'AUTH DENY'
        
        ans = model.predict(ques, uid)

        self.write(ans)


def make_app():
    return tornado.web.Application([
        (r"/botchat", MainHandler, {}, 'botchat'),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8899)
    tornado.ioloop.IOLoop.current().start()
