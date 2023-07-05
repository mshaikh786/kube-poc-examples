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
                '--password', '--uid=%s'%uid, 
                '--gidnumber=%d'%gid, 
                '--first=%s'%First,'--last=%s'%Last,
                '--homedir=/home/%s'%uid,
                '--email=%s'%email] )
            i+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',type=str,help='CSV filename with information about users') 
    args=parser.parse_args()
    create_users(args.f)