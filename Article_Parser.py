import os
import re
import praw
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import html2text


# for sub in subs:
#   res = requests.get(sub.url)
#   if (res.status_code == 200 and 'content-type' in res.headers and
#       res.headers.get('content-type').startswith('text/html')):
#     html = res.text

def get_article(url):
    print('  - Retrieving %s' % url)
    try:
        res = requests.get(url)
        if (res.status_code == 200 and 'content-type' in res.headers and
                res.headers.get('content-type').startswith('text/html')):
            article = parse(res.text)
            # print('      => Title = "%s"' % article['title'])
            print('      => Content = "%s"' % article['content'])
            return article
        else:
            print('      x fail or not html')
    except Exception as e:
        print('Some exception occured.')
        print(e)
        pass


def parse(text):
    soup = BeautifulSoup(text, 'html.parser')
    for div in soup.find_all('div'):
        if div.text.find(" ") == -1:
            div.decompose()
        if len(div.text) < 300:
            div.decompose()
    for nav in soup.find_all('nav'):
        nav.decompose()
    # for span in soup.find_all('span'):
    #     span.decompose()
    for img in soup.find_all('img'):
        img.decompose()
    for ul in soup.find_all('ul'):
        ul.decompose()
    for li in soup.find_all('li'):
        li.decompose()
    for h2 in soup.find_all('h2'):
        h2.decompose()
    for h3 in soup.find_all('h3'):
        h3.decompose()
    for h4 in soup.find_all('h4'):
        h4.decompose()
    for h5 in soup.find_all('h5'):
        h5.decompose()
    for h6 in soup.find_all('h6'):
        h6.decompose()
    elems = soup.find_all(class_=True)
    for elem in elems:
        try:
            # if elem.text.find(" ") == -1:
            #     elem.decompose()
            if len(elem.text) < 50:
                elem.decompose()
            elif elem.has_attr("class"):
                for clas in elem["class"]:
                    if "footer" in clas:
                        for child in elem.find_all():
                            child.decompose()
        except Exception as e:
            elem.decompose()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.body_width = 0
    # h.use_automatic_links = False
    h.ignore_images = True
    formatted_text = h.handle(str(soup))
    return  {'title': "test", 'content': formatted_text}


def parse_article(text):
    soup = BeautifulSoup(text, 'html.parser')

    text = ''.join(BeautifulSoup(text, "html.parser").stripped_strings)
    print(text)
    # find the article title
    h1 = soup.body.find('h1')

    # find the common parent for <h1> and all <p>s.
    root = h1
    while root.name != 'body' and len(root.find_all('p')) < 5:
        root = root.parent

    if len(root.find_all('p')) < 1:
        return None

    # find all the content elements.
    ps = root.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'div'])
    ps.insert(0, h1)
    content = [tag2md(p) for p in ps]
    print(content)
    return {'title': h1.text, 'content': content}


def tag2md(tag):
    print("tag2md")
    print("Length is %d" % len(tag.text))
    if len(tag.text) < 100:
        return ""
    if tag.name == 'p':
        return tag.text
    elif tag.name == 'h1':
        return f'{tag.text}\n{"=" * len(tag.text)}'
    elif tag.name == 'h2':
        return f'{tag.text}\n{"-" * len(tag.text)}'
    elif tag.name in ['h3', 'h4', 'h5', 'h6']:
        return f'{"#" * int(tag.name[1:])} {tag.text}'
    elif tag.name == 'pre':
        return f'```\n{tag.text}\n```'
    elif tag.name == 'div':
        if (len(tag.text)) > 1000:
            return f'```\n{tag.text}\n```'
        else:
            return ""
    else:
        return ""


def main():
    articles = [
        "http://www.catchnews.com/national-news/bihar-bjp-and-jdu-workers-clash-over-ram-temple-issue-creates-furor-over-poll-manifesto-156689.html"]

    for article in articles:
        print('Scraping /r/%s...' % article)
        get_article(article)


if __name__ == '__main__':
	main()

# t = get_article(' https://theprint.in/science/himalayan-fossil-of-adolescents-jaw-may-explain-how-tibetans-sherpas-master-mountains/230275/');
# print(t["article"])

