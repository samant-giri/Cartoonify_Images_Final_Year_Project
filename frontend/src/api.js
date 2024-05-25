import axios from "axios";

const baseUrl = "http://localhost:5000/cartoonify";

function transform(data) {
  return axios.post(baseUrl, data);
}

export { transform };
