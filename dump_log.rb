# coding: utf-8
require "nokogiri"
require "open-uri"
require "json"

# ruby script/dump_log.rb 2017-01-02
#!/usr/bin/ruby

date = ARGV[0].dup.force_encoding("utf-8")

log = Nokogiri::HTML(open("http://ni.singulariti.io:32419/stats/dump-stats-by-day?start=" + date + "&sig=qdjzni"), nil, "UTF-8")

log_json = JSON.parse(log);

File.open("data.json","w") do |f|
  f.write(JSON.pretty_generate(log_json))

end
#print JSON.pretty_generate(log_json);
