import MySQLdb as mdb
from MySQLdb.cursors import SSDictCursor
import json


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-output', default='/data/home/cul226/wos_dump')
    args = argparser.parse_args()

    conn = mdb.connect(host='heisenberg', user='wos-user', passwd='#uHbG9LA',
                       db='wos_tiny')
    cur = conn.cursor(SSDictCursor)
    query = 'select uid, title, abstract from papers where abstract is not NULL'
    cur.execute(query)

    cnt = 0
    with open(args.output, 'w') as out:
        for row in cur:
            if cnt % 10000 == 0:
                print '{}0k'.format(cnt / 10000) if cnt != 0 else 0
            out.write(json.dumps(row) + '\n')
            cnt += 1

