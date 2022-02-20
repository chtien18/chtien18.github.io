---
#
# Use the widgets beneath and the content will be
# inserted automagically in the webpage. To make
# this work, you have to use › layout: frontpage
#
layout: frontpage
header:
  image_fullwidth: cropped-samurai.jpg
widget1:
  title: "Documentation"
  url: ''
  image: widget-datsach.jpg
  text: '...'

widget2:
  title: "Learn"
  url: 'https://chtien18.github.io/learn/'
  image: widget-phanbon.jpg
  text: '...'
#  video: '<a href="#" data-reveal-id="videoModal"><img src="/images/" width="302" height="182" alt=""/></a>'
widget3:
  title: "Family"
  url: ''
  image: widget-tinhdausa.jpg
  text: ''
  
#
# Use the call for action to show a button on the frontpage
#
# To make internal links, just use a permalink like this
# url: /getting-started/
#
# To style the button in different colors, use no value
# to use the main color or success, alert or secondary.
# To change colors see sass/_01_settings_colors.scss
#
callforaction:
  url: https://eitalab.com/category/khoa-hoc/
  text: Read more.
  style: alert
permalink: /index.html
#
# This is a nasty hack to make the navigation highlight
# this page as active in the topbar navigation
#
homepage: true
---
