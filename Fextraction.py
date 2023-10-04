

    
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from  nltk.tokenize import word_tokenize
import nltk 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import textdistance
import os
from bs4 import BeautifulSoup
import mailparser
from collections import OrderedDict
import email
from urllib.request import urlretrieve
import tarfile
import shutil
import numpy as np
import glob
import mailparser
import re
import tldextract
import urllib.request
import warnings
import csv
import sys
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import json
import ast
import re

import mailparser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
class FeatureExtraction:
 def get_URLs(mail_body):
        clean_payload = re.sub(r'\s+', ' ', mail_body)
        soup = BeautifulSoup(clean_payload, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a')]
        result = [link for link in links if isinstance(link, (str, bytes)) and is_URL(link)]
        urlregex = re.compile(URLREGEX_NOT_ALONE, re.IGNORECASE)
        result += urlregex.findall(clean_payload)
        result = list(set(result))
        return result
 
 def load_mails(dirpath):
    """load emails from the specified directory"""

    return [open(f, 'r', encoding='utf-8', errors='ignore').read() for f in glob.glob(f'{dirpath}/*')]   

      
 def extract(mail):
    parsed_mail = getMailBody(mail)    
                        
    mail_body = parsed_mail[0]
    
        
    parsed_mail = getMailBody(mail)
    urls = get_URLs(mail_body)
    mail_headers = parsed_mail[2]
    mail_subject = parsed_mail[1]       
    features =[]
        
    
    
    
    
    



  


    features.append( int(presenceHTML(mail)==True))
        
    features.append(int(presenceHTMLFORM(mail_body)==True))
    features.append( int(presenceHTMLIFRAME(mail_body)==True))
        
    features.append(int(presenceFlashContent(mail_body)==True))
        
    features.append( int(presenceGeneralSalutation(mail_body)==True))
        
    features.append(int(presenceJavaScript(mail_body)==True))
        
    features.append(int(mail_to(mail_body)==True))
        
    features.append(popups(mail_body))
        
    features.append(body_richness(mail_body))
        
    features.append(len(urls))
        
    features.append((malicious_URL(urls)))
        
    features.append(text_link_disparity(mail_body))
        
    features.append (numberOfAttachments(mail))
        
    features.append ((IP_as_URL(urls)))
        
    features.append ((hexadecimal_URL(urls)))
        
    features.append (int(presence_bad_ranked_URL(urls)==True))
        
    features.append ((max_domains_counts(urls)))
        
    features.append(at_in_URL(urls))
        
    features.append (subject_richness(mail_subject))
        
    features.append(int(isForwardedMail(mail_subject)==True))
        
    features.append(int(isRepliedMail(mail_subject)==True))
        
    features.append(int(contains_account(mail_subject)== True))
        
    features.append(int(contains_verify(mail_subject)==True))
        
    features.append(int(contains_update(mail_subject)==True))

    features.append(int(contains_prime_targets(mail_subject)==True))
       
    features.append(int(contains_suspended(mail_subject)==True))
      
    features.append(int(contains_password(mail_subject)==True))
       
    features.append(int(contains_urgent(mail_subject)==True))
       
    features.append (int(contains_access(mail_subject)==True))
        
    features.append( number_of_dots(mail_headers))
       
    features.append(number_of_dash(mail_headers))
        
       

    
    return features  
       









    

URLREGEX = r"^(https?|ftp)://[^\s/$.?#].[^\s]*$"
URLREGEX_NOT_ALONE = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
FLASH_LINKED_CONTENT = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])+).*\.swf"
HREFREGEX = '<a\s*href=[\'|"](.*?)[\'"].*?\s*>'
IPREGEX = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))\b"
MALICIOUS_IP_URL = r"\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\/(www|http|https|ftp))\b"
EMAILREGEX = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
GENERAL_SALUTATION = r'\b(dear|hello|Good|Greetings)(?:\W+\w+){0,6}?\W+(user|customer|seller|buyer|account holder)\b'

stop_words = set(stopwords.words('english'))



def getMailBody(mail):
        try:
            parsed_mail = mailparser.parse_from_string(mail)
            mail_body = parsed_mail.body.lower()
            subject = parsed_mail.subject
            headers = parsed_mail.headers
            
        except UnicodeDecodeError as Argument:
            parsed_mail = email.message_from_string(mail)
            body = ""
            if parsed_mail.is_multipart():
                for part in parsed_mail.walk():
                    # returns a bytes object
                    payload = part.get_payload(decode=True)
                    strtext = payload.decode()
                    body += strtext
            else:
                payload = parsed_mail.get_payload(decode=True)
                strtext = payload.decode()
                body += strtext
            headers = email.parser.HeaderParser().parsestr(mail)
            mail_body = body.lower()
            subject = headers['Subject']
        return [mail_body,subject,headers]



   



def cleanhtml(sentence):
        return re.sub('<.*?>', ' ', sentence)

def cleanpunc(sentence):
        cleaned = re.sub(r'[?|!|\'|"|#]', '', sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]', ' ', cleaned)
        return cleaned

def cleanBody(mail_body):
        filtered_text = cleanpunc(cleanhtml(mail_body))
        word_tokens = word_tokenize(filtered_text)
        filtered = filter(lambda w: w not in stop_words and w.isalpha(), word_tokens)
        return list(filtered)

presenceHTML = lambda mail: int((email.message_from_string(mail).get_content_type() == 'text/html') == True)

presenceHTMLFORM = lambda message: int((re.compile(r'<\s?\/?\s?form\s?>', re.IGNORECASE).search(message) != None) == True)

presenceHTMLIFRAME = lambda message: int(re.compile(r'<\s?\/?\s?iframe\s?>', re.IGNORECASE).search(message) != None) == True

presenceJavaScript = lambda message: int(re.compile(r'<\s?\/?\s?script\s?>', re.IGNORECASE).search(message) != None) == True

presenceFlashContent = lambda message: int(len(re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])+).*\.swf", message, re.IGNORECASE)) > 0 or (re.compile(r'embed\s*src\s*=\s*\".*\.swf\"', re.IGNORECASE).search(message) != None)) == True

presenceGeneralSalutation = lambda message: int(re.compile(GENERAL_SALUTATION, re.IGNORECASE).search(message) != None) == True

def numberOfAttachments(raw_mail):
        try:
            mail = mailparser.parse_from_string(raw_mail)
            count = len(mail.attachments)
            return count
        except:
            return 0

mail_to = lambda mail_body: int(re.compile(r'mailto:', re.IGNORECASE).search(mail_body) != None) == True

popups = lambda mail_body: 1 if re.compile(r'window.open|onclick', re.IGNORECASE).search(mail_body) else 0

def body_richness(mail_body):
        mail_body = cleanBody(mail_body)
        return len(mail_body) / len(set(mail_body)) if set(mail_body) else len(mail_body)

def is_URL(link):
        return bool(re.search(URLREGEX, link, re.IGNORECASE))

def get_URLs(mail_body):
        clean_payload = re.sub(r'\s+', ' ', mail_body)
        soup = BeautifulSoup(clean_payload, 'html.parser')
        links = [link.get('href') for link in soup.find_all('a')]
        result = [link for link in links if isinstance(link, (str, bytes)) and is_URL(link)]
        urlregex = re.compile(URLREGEX_NOT_ALONE, re.IGNORECASE)
        result += urlregex.findall(clean_payload)
        result = list(set(result))
        return result
    
def load_mails(dirpath):
    """load emails from the specified directory"""

    return [open(f, 'r', encoding='utf-8', errors='ignore').read() for f in glob.glob(f'{dirpath}/*')]    

def loadd_mails(f):
    """load emails from the specified directory"""

    return [open(f, 'r', encoding='utf-8', errors='ignore').read() ]    



def IP_as_URL(urls):
        result = []
        for url in urls:
            match = re.search(IPREGEX, url, re.IGNORECASE)
            if match and match.group(1):
                result.append(match.group(1))
        return len(result)

def text_link_disparity(mail_body):
        soup = BeautifulSoup(mail_body, 'html.parser')
        count = sum(1 for item in soup.find_all('a') 
                    for string in item.stripped_strings
                    if is_URL(string) and string != item.get('href'))
        return count

def malicious_URL(urls):
        count = sum(1 for url in urls
                    if (re.search(IPREGEX, url, re.IGNORECASE) or
                        len(re.findall(r'(https?://)', url, re.IGNORECASE)) > 1 or
                        len(re.findall(r'(www\.)', url, re.IGNORECASE)) > 1 or
                        len(re.findall(r'(\.com|\.org|\.co)', url, re.IGNORECASE)) > 1))
        return count

def hexadecimal_URL(urls):
        count = sum(1 for url in urls
                    if re.search(r'%[0-9a-fA-F]+', url, re.IGNORECASE))
        return count




alexa_rank_cache = {}
def get_Alexa_Rank(domain):
        if domain in alexa_rank_cache:
            return int(alexa_rank_cache[domain])
        try:
            xml = urllib.request.urlopen(
                'http://data.alexa.com/data?cli=10&dat=s&url=%s' % domain).read().decode('utf-8')
            rank = int(re.findall(r'RANK="(\d+)"', xml, re.IGNORECASE)[1])
        except:
            rank = -1
        alexa_rank_cache[domain] = rank
        return rank

def extract_domains(urls):
        domain_set = set(tldextract.extract(url).registered_domain for url in urls)
        return list(domain_set)

def domain_counts(url):
        domains = tldextract.extract(url)
        count = len(re.findall(r'\.', domains.subdomain, re.IGNORECASE)) + \
                len(re.findall(r'\.', domains.domain, re.IGNORECASE)) + 1
        if re.search(IPREGEX, domains.domain, re.IGNORECASE):
            count -= 3
        return count

def presence_bad_ranked_URL(urls):
        domains = extract_domains(urls)
        max_rank = 0
        for domain in domains:
            rank = get_Alexa_Rank(domain)
            max_rank = max(rank, max_rank)
            if rank == -1:
                return 0
        if max_rank > 70000:
            return 1
        return 0

def max_domains_counts(urls):
        count = 1
        for url in urls:
            count = max(domain_counts(url), count)
        return count


def at_in_URL(urls):
        for url in urls:
            if (re.compile(r'@',re.IGNORECASE).search(url)):
                return 1
            else: 
                continue
        return 0
    
def write_cache():
        with open('./cache/alexa_rank_cache.txt', 'w') as cache_file:
            json.dump(alexa_rank_cache, cache_file)

def loadCache():
            with open('./cache/alexa_rank_cache.txt','r') as cache_file:
                cache = ast.literal_eval(cache_file.read())
                alexa_rank_cache = cache



   

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def purify(subject):
        filtered = ""
        word_tokens = word_tokenize(subject)
        for w in word_tokens:
            if w not in stop_words and w.isalpha():
                    w = stemmer.stem(w)
                    filtered+=(lemmatizer.lemmatize(w))
                    filtered+=" "
        return filtered

def contains_word(subject, word):
        subject = purify(subject)
        jaro = textdistance.Jaro()
        for w in subject.split():
            if jaro(word, w) > 0.9:
                return 1
        return 0

def isRepliedMail(subject):
        return subject.startswith('Re:')

def isForwardedMail(subject):
        return subject.startswith('Fwd:')

def subject_richness(subject):
        texts = subject.split()
        if len(set(texts)) != 0:
            return len(texts) / len(set(texts))
        else:
            return len(texts)

def contains_verify(subject):
        return contains_word(subject, 'verify')

def contains_update(subject):
        return contains_word(subject, 'update')

def contains_access(subject):
     return contains_word(subject, 'access')

def contains_prime_targets(subject):
        return (contains_word(subject, 'bank') or
                contains_word(subject, 'Paypal') or
                contains_word(subject, 'ebay') or
                contains_word(subject, 'amazon'))

def contains_account(subject):
        return (contains_word(subject, 'account') or
                contains_word(subject, 'profile') or
                contains_word(subject, 'handle'))

def contains_suspended(subject):
        return (contains_word(subject, 'closed') or
                contains_word(subject, 'expiration') or
                contains_word(subject, 'suspended') or
                contains_word(subject, 'terminate') or
                contains_word(subject, 'restricted'))

def contains_password(subject):
        return (contains_word(subject, 'password') or
                contains_word(subject, 'credential'))

def contains_urgent(subject):
        return (contains_word(subject, 'urgent') or
                contains_word(subject, 'immediate'))

def number_of_dots(headers):
    sender = headers.get("from")
    if sender is None:
        # Handle case where "from" key is missing
        return 0
    else:
        return sender.count('.')


def number_of_dash(headers):
    sender = headers.get("from")
    if sender is None:
        # Handle case where "from" key is missing
        return 0
    else:
        return sender.count('-')






