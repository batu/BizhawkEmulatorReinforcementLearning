print("Started LUA code.")
io.stdout:write('start ' .. gameinfo.getromname() .. '\n')

print("Define serialize")
function serialize(t)
  local serializedValues = {}
  local value, serializedValue
  for i=1,#t do
    value = t[i]
    serializedValue = type(value)=='table' and serialize(value) or value
    table.insert(serializedValues, serializedValue)
  end
  return string.format("{ %s }", table.concat(serializedValues, ', ') )
end


i = 0
buttons = {}
while true do
	if i > 10 then
		temp = io.stdin:read("*a")
		loadstring(temp)()
		io.stdout:write("\n")
	else
		i = i + 1
		emu.frameadvance()
	end
end
