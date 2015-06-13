# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:25:32 2014

@author: Edward
"""

import mechanize, urllib2, cookielib, random, time

class anonBrowser(mechanize.Browser):
    def __init__(self, proxies = [], user_agents = []):
        mechanize.Browser.__init__(self)
        self.set_handle_robots(False)
        self.proxies = proxies
        self.user_agents = user_agents + ['Mozilla/12.0', 'FireFox/24.0',
                                          'ExactSearch','iOS8.1']
        self.cookie_jar = cookielib.LWPCookieJar()
        self.set_cookiejar(self.cookie_jar)
        self.anonymize()
    def clear_cookies(self):
        self.cookie_jar = cookielib.LWPCookieJar()
        self.set_cookiejar(self.cookie_jar)
    def change_user_agent(self):
        index = random.randrange(0, len(self.user_agents))
        self.addheaders = [('User-agent', (self.user_agents[index]))]
    def change_proxy(self):
        if self.proxies:
            index = random.randrange(0, len(self.prproxies))
            self.set_proxies({'http': self.proxies[index]})
    def anonymize(self, sleep=False):
        self.clear_cookies()
        self.change_user_agent()
        self.change_proxy()
        if sleep:
            time.sleep(60)
    
def viewOnlineText(text_url):
    for line in urllib2.urlopen(text_url):
        return line.strip('\n\t .,;')
        break

ab = anonBrowser(proxies=[], user_agents=[('User-agent','superSecretBrowser')])
for attempt in range(1,5):
    ab.anonymize()
    print '[*] Fetching page ...'
    response = ab.open('http://kittenwar.com')
    for cookie in ab.cookie_jar:
        print cookie
