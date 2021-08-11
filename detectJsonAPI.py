from wsgiref.simple_server import make_server
from urllib.parse import parse_qs


def application(environ, start_response):
    print(environ['PATH_INFO'])
    start_response('200 OK', [('Content-Type', 'text/html;charset=utf-8')])
    params = parse_qs(environ['QUERY_STRING'])
    params_text = str(params.get('txt', [''])[0])
    body = "hello" + params_text
    return [body.encode('utf-8')]


if __name__ == '__main__':
    httpd = make_server('', 8000, application)
    print('Serving HTTP on port 8000...')
    httpd.serve_forever()
