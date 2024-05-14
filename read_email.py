import email

def extract_body_from_eml(file_path):
    """
    从.eml文件中提取邮件正文。
    
    参数:
    file_path (str): .eml文件的路径。
    
    返回:
    str: 邮件的正文内容。
    """
    # 打开并解析邮件文件
    with open(file_path, 'rb') as file:
        msg = email.message_from_binary_file(file)
        
    # 初始化正文为空字符串
    body = ""
    
    # 遍历邮件的各个部分，寻找正文
    if msg.is_multipart():
        # 如果邮件有多部分，遍历每部分
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # 忽略附件和其他非文本部分
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                # 解码并加入正文
                body = part.get_payload(decode=True).decode(part.get_content_charset(), errors="ignore")
                break  # 停止循环，因为我们只关心第一个纯文本部分
    else:
        # 如果邮件不是多部分，则直接提取正文
        body = msg.get_payload(decode=True).decode(msg.get_content_charset(), errors="ignore")

    return body

# 示例用法
# script_dir = os.path.dirname(os.path.abspath(__file__))
# email_file_path = os.path.join(script_dir, 'test.eml')
# email_body = extract_body_from_eml(email_file_path)
# print(email_body)