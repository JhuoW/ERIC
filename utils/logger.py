import time

starttime = time.time()

def my_clock():
	'''basically time.clock()
	'''
	return time.time() - starttime
def len_ignore_n(s):
	'''len(s) ignoring \\n
	'''
	s = str(s).strip()
	s = s.replace("\n" , "")

	l = (len(bytes(s , encoding = "utf-8")) - len(s)) // 2 + len(s) #中文
	l += 7 * s.count("\t")											#\t

	return l

def last_len(s):
	'''length of last line
	'''
	s = str(s).strip()
	return len_ignore_n(s.split("\n")[-1])

class Logger:
	'''auto log
	'''
	def __init__(self , mode = [print] , log_path = None , append = ["clock"] , line_length = 90):
		if log_path:
			self.log_fil = open(log_path , "w" , encoding = "utf-8")
		else:
			self.log_fil = None

		self.mode = mode

		if ("write" in mode) and (not log_path):
			raise Exception("Should have a log_path")

		self.append 	= append
		self.line_length = line_length

	def close(self):
		if self.log_fil:
			self.log_fil.close()

	def log(self , content = ""):

		content = self.pre_process(content)

		for x in self.mode:
			if x == "write":
				self.log_fil.write(content + "\n")
				self.log_fil.flush()
			else:
				x(str(content))

	def add_line(self , num = -1 , char = "-"):
		if num < 0:
			num = self.line_length
		self.log(char * num)

	def pre_process(self , content):
		insert_space = self.line_length - last_len(content) #complement to line_length 
		content += " " * insert_space

		for x in self.append: #add something to the end

			y = ""
			if x == "clock":
				y = "%.2fs" % (my_clock())
			elif x == "time":
				y = time.strftime("%Y-%m-%d %H:%M:%S" , time.localtime() )
			else:
				y = x()

			content += "| " + y + " "


		return content