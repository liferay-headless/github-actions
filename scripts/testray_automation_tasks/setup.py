# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.md') as f:
    readme = f.read()

with open('utils/LICENSE') as f:
    license_at = f.read()

setup(
    name='TestrayAutomatedTasks',
    version='0.1.0',
    description='Package to do perform some automated task for orchestration of Liferay tasks in Testray and Jira',
    long_description=readme,
    author='Magdalena Jedraszak',
    author_email='magdalena.jedraszak@liferay.com',
    url='https://github.com/magjed4289/TestrayAutomatedTasks',
    license=license_at,
    packages=['liferay']
)
