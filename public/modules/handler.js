class Handler {
  resized () {
    let width = document.body.clientWidth;
    let cardwidth = 200;
    let leftover = width % cardwidth;
    let count = Math.floor(width / cardwidth)
    let new_cardwidth = "" + (cardwidth + leftover / count - 2) + "px"
    document.body.style.setProperty("--card-width", new_cardwidth)
  }
  constructor (app) {
    this.app = app
    let id;
    window.addEventListener('resize', () => {
      clearTimeout(id);
      id = setTimeout(this.resized, 500);
    });

    this.resized()
    document.querySelector(".container").addEventListener("click", async (e) => {
      e.preventDefault()
      e.stopPropagation()
      let colTarget = (e.target.classList.contains(".col") ? e.target : e.target.closest(".col"))
      let fullscreenTarget = (e.target.classList.contains(".gofullscreen") ? e.target : e.target.closest(".gofullscreen"))
      let clipboardTarget = (e.target.classList.contains(".copy-text") ? e.target : e.target.closest(".copy-text"))
      let tokenTarget = (e.target.classList.contains(".token") ? e.target : e.target.closest(".token"))
      let tokenPopupTarget = (e.target.classList.contains(".popup-link") ? e.target : e.target.closest(".popup-link"))
      let grabTarget = (e.target.classList.contains(".grab") ? e.target : e.target.closest(".grab"))
      let openFileTarget = (e.target.classList.contains(".open-file") ? e.target : e.target.closest(".open-file"))
      let displayMetaTarget = (e.target.classList.contains(".view-xmp") ? e.target : e.target.closest(".view-xmp"))
      let favoriteFileTarget = (e.target.classList.contains(".favorite-file") ? e.target : e.target.closest(".favorite-file"))
      let card = (e.target.classList.contains("card") ? e.target : e.target.closest(".card"))
      if (card) card.classList.remove("fullscreen")
      if (fullscreenTarget) {
        let img = fullscreenTarget.closest(".card").querySelector("img").cloneNode()
        let scaleFactor = Math.min(window.innerWidth / img.naturalWidth, window.innerHeight / img.naturalHeight)
        if (this.viewer) this.viewer.destroy()
        this.viewer = new Viewer(img, {
          transition: false,
          toolbar: {
            'zoomIn': true,
            'zoomOut': true,
            'reset': true,
            'play': true,
            'oneToOne': true,
            'rotateLeft': true,
            'rotateRight': true,
            'flipHorizontal': true,
            'flipVertical': true
          },
          viewed() {
            this.viewer.zoomTo(scaleFactor)
          },
        });
        this.viewer.show()
      } else if (openFileTarget) {
        window.electronAPI.open(openFileTarget.getAttribute("data-src"))
      } else if (favoriteFileTarget) {
        let data_favorited = favoriteFileTarget.getAttribute("data-favorited")
        let is_favorited = (data_favorited === "true" ? true : false)
        let src = favoriteFileTarget.getAttribute("data-src")
        let root = favoriteFileTarget.closest(".card").querySelector("img").getAttribute("data-root")
        let favClass
        if (is_favorited) {
          // unfavorite
          let response = await window.electronAPI.gm({
            cmd: "set",
            args: [
              src,
              [{
                key: "dc:subject",
                val: ["favorite"],
                mode: "delete"
              }]
            ]
          })
          favoriteFileTarget.setAttribute("data-favorited", "false")
          favClass = "fa-regular fa-heart"

          // remove 'favorite' tag from the tags area
          favoriteFileTarget.closest(".card").querySelector("tr[data-key=tags] td span[data-tag='tag:favorite']").remove()
        } else {
          // favorite
          let response = await window.electronAPI.gm({
            cmd: "set",
            args: [
              src,
              [{
                key: "dc:subject",
                val: ["favorite"],
                mode: "merge"
              }]
            ]
          })
          favoriteFileTarget.setAttribute("data-favorited", "true")
          favClass = "fa-solid fa-heart"

          // add 'favorite' tag
          let span = document.createElement("span")
          span.setAttribute("data-tag", "tag:favorite")
          span.innerHTML = `<button data-tag="tag:favorite" class='tag-item'><i class="fa-solid fa-tag"></i> favorite</button>`
          favoriteFileTarget.closest(".card").querySelector("tr[data-key=tags] td.attr-val").appendChild(span)
        }
        favoriteFileTarget.querySelector("i").className = favClass

        await this.app.synchronize([{ file_path: src, root_path: root }], async () => {
          document.querySelector(".status").innerHTML = ""
          document.querySelector("#sync").classList.remove("disabled")
          document.querySelector("#sync").disabled = false
          document.querySelector("#sync i").classList.remove("fa-spin")
        })
      } else if (grabTarget) {
      } else if (tokenTarget && e.target.closest(".card.expanded")) {
        let key = tokenTarget.closest("tr").getAttribute("data-key")
        let val = tokenTarget.getAttribute("data-value")

        let popup_items = []
        if (key === "file_path" || key === "model_name" || key === "agent") {
          if (val.split(" ").length > 1) {
            val = `"${val}"`
          }
          if (key === "file_path") {
            popup_items = [
              `<span class='popup-link' data-key='${key}' data-value='${val}'>${val}</span>`,
              `<span class='popup-link' data-key='-${key}' data-value='${val}'><i class="fa-solid fa-not-equal"></i> ${val}</span>`
            ]
          }
        }
        if (key === "caption" || key === "has_caption" || key === "captioned_by") {
          val = `"${val}"`
          popup_items = [
            `<span class='popup-link' data-key='${key}' data-value='${val}'>${val}</span>`,
            `<span class='popup-link' data-key='-${key}' data-value='${val}'><i class="fa-solid fa-not-equal"></i> ${val}</span>`
          ]
        }
        if (key === "prompt") {
          popup_items = [
            `<span class='popup-link' data-key='${key}' data-value='${val}'>${val}</span>`,
            `<span class='popup-link' data-key='-${key}' data-value='${val}'><i class="fa-solid fa-not-equal"></i> ${val}</span>`
          ]
        }

        if (key === "tags") {
          console.log({ key , val })
          if (val.split(" ").length > 1) {
            val = val.replace(/^tag:(.+)/, 'tag:"$1"')
          }
          popup_items = [
            `<span class='popup-link' data-key='prompt' data-value='${val}'>${val}</span>`,
            `<span class='popup-link' data-key='-prompt' data-value='-${val}'><i class="fa-solid fa-not-equal"></i> ${val}</span>`
          ]
        }

        if (key === "width" || key === "height") {
          popup_items = [
            `<span class='popup-link' data-key='-${key}' data-value='${val}'>&lt;</span>`,
            `<span class='popup-link' data-key='-=${key}' data-value='${val}'>&lt;=</span>`,
            `<span class='popup-link' data-key='${key}' data-value='${val}'>${val}</span>`,
            `<span class='popup-link' data-key='+=${key}' data-value='${val}'>=&gt;</span>`,
            `<span class='popup-link' data-key='+${key}' data-value='${val}'>&gt;</span>`
          ]
        }

        if (popup_items.length > 0) {
          tippy(tokenTarget, {
            interactive: true,
  //          placement: "bottom-end",
            trigger: 'click',
            content: `<div class='token-popup'>${popup_items.join("")}</div>`,
            allowHTML: true,
          }).show();
        } else {
          this.app.navbar.input(key, val)
        }


      } else if (tokenPopupTarget) {
        let key = tokenPopupTarget.getAttribute("data-key")
        let val = tokenPopupTarget.getAttribute("data-value")
        console.log({ key , val })
        this.app.navbar.input(key, val)
      } else if (displayMetaTarget) {
        let file_path = displayMetaTarget.getAttribute("data-src")
        let xml = await window.electronAPI.xmp(file_path)
        let textarea = displayMetaTarget.closest(".xmp").querySelector(".slot")
        textarea.classList.toggle("hidden")
        textarea.value = xml;
        textarea.style.height = "" + (textarea.scrollHeight + 2) + "px";
      } else if (colTarget && e.target.closest(".card.expanded")) {
        // if clicked inside the .col section when NOT expanded, don't do anything.
        // except the clipboard button
        // if the clicked element is the delete button, delete
        if (clipboardTarget) {
          window.electronAPI.copy(clipboardTarget.getAttribute("data-value"))
          clipboardTarget.querySelector("i").classList.remove("fa-regular")
          clipboardTarget.querySelector("i").classList.remove("fa-clone")
          clipboardTarget.querySelector("i").classList.add("fa-solid")
          clipboardTarget.querySelector("i").classList.add("fa-check")
          clipboardTarget.querySelector("span").innerHTML = "copied"

          setTimeout(() => {
            clipboardTarget.querySelector("i").classList.remove("fa-solid")
            clipboardTarget.querySelector("i").classList.remove("fa-check")
            clipboardTarget.querySelector("i").classList.add("fa-regular")
            clipboardTarget.querySelector("i").classList.add("fa-clone")
            clipboardTarget.querySelector("span").innerHTML = "copy"
          }, 1000)
        }
      } else {
        let target = (e.target.classList.contains("card") ? e.target : e.target.closest(".card"))
        if (target) {
          target.classList.toggle("expanded")
          if (target.classList.contains("expanded")) {
            let img = target.querySelector("img").cloneNode()
            let scaleFactor = Math.min(window.innerWidth / img.naturalWidth, window.innerHeight / img.naturalHeight)
            if (this.viewer) this.viewer.destroy()
            this.viewer = new Viewer(img, {
              transition: false,
              viewed() {
                this.viewer.zoomTo(scaleFactor)
              },
            });
          }
        }
      }
    })
  }
}
