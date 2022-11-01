function appendOutput(data) {
  data = data.reverse();
  for (let i = 0; i < data.length; i++) {
    let outputNode = document.createElement("figure");
    const htmlObject = `
    Output:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Init Image:
    </br>
    <img src="data:image/png;base64,${data[i][1]}"
         alt="${"result_image"}"
         title="${"result_image"}"
         loading="lazy"
         width="256"
         height="256">
    </a>
    <img src="data:image/png;base64,${data[i][3]}"
         alt="${"init_image"}"
         title="${"init_image"}"
         loading="lazy"
         width="256"
         height="256">
    </a>
    </br>
    type: ${data[i][0]}
    </br>
    prompt: ${data[i][2]}
    </br>
    strength: ${data[i][4]}
    </br>
    iterations: ${data[i][5]}
    </br>
    steps: ${data[i][6]}
    </br>
    size: ${data[i][7]} x ${data[i][8]}
    </br>
    seamless: ${data[i][9]}
    </br>
    fit: ${data[i][10]}
    </br>
    mask: ${data[i][11]}
    </br>
    invert_mask: ${data[i][12]}
    </br>
    cfg_scale: ${data[i][13]}
    </br>
    sampler_name: ${data[i][14]}
    </br>
    gfpgan_strength: ${data[i][15]}
    </br>
    upscale: ${data[i][16]}
    </br>
    progress_images: ${data[i][17]}
    </br>
    seed: ${data[i][18]}
    </br>
    variation_amount: ${data[i][19]}
    </br>
    with_variations: ${data[i][20]}
    </br>
    date: ${data[i][21]}
    `;
    outputNode.innerHTML = htmlObject;
    document.querySelector("#results").prepend(outputNode);
  }
}

const BLANK_IMAGE_URL =
  'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"/>';
async function generateSubmit(form) {
  const prompt = document.querySelector("#prompt").value;
  const count = parseInt(prompt);
  document.querySelector("#results").innerHTML = "";

  fetch("/load_db?count=" + count, {
    method: "GET",
  }).then(async (response) => {
    const reader = response.body.getReader();

    let jsonData = "";
    while (true) {
      let { value, done } = await reader.read();
      value = new TextDecoder().decode(value);
      jsonData += value;
      if (done) {
        const data = JSON.parse(jsonData);
        appendOutput(data);
        break;
      }
    }
  });
}

window.onload = async () => {
  document.querySelector("#prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      const form = e.target.form;
      generateSubmit(form);
    }
  });
  document.querySelector("#generate-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const form = e.target;

    generateSubmit(form);
  });
};
