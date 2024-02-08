#!/usr/bin/env python3
import csv,os
import argparse, subprocess as sb
def create_users(filename):
    with open(filename,newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        base_uid=800000
        i=0
        for row in reader:
            if (i>0):
                First=row[0]
                Last=row[1]
                username=row[2]
                uid=base_uid+i
                gid=uid
                password=row[3]
                email=row[4]
           
                sb.run(['ipa','user-add','%s' %username,
                '--uid=%s'%uid, 
                '--gidnumber=%d'%gid, 
                '--first=%s'%First,'--last=%s'%Last,
                '--homedir=/home/%s'%uid,
                '--email=%s'%email] )
                sb.run(['ipa','passwd','%s' %username,'%s'%password] )

            i+=1
def delete_user(username):
    sb.run(['ipa','user-del','%s'%username])
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',type=str,help='CSV filename with information about users') 
    parser.add_argument('-d',type=str,help='delete user') 
    args=parser.parse_args()
    if args.f is not None:
        create_users(args.f)
    elif args.d is not None:
        delete_user(args.d)
