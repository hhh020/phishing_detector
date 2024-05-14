# phishing_detector
基于LSTM的钓鱼邮件检测,用于提取邮件数据并进行训练,得到LSTM模型

Phishing_Email.csv --数据集
training.py --训练模型
test.py --测试
read_email.py 读取邮件
phishing_detector.py --集成的检测系统

使用命令
`python3 phishing_detector.py {email_path}`
即可检测邮件是否为钓鱼邮件
