import React from "react";
import {
  FaHome,
  FaChartBar,
  FaMap,
  FaQuestionCircle,
  FaChevronRight,
  FaPlayCircle,
} from "react-icons/fa";
import { useNavigate } from "react-router-dom";

function Faq() {
  const navigate = useNavigate();

  const handleClickHome = () => {
    navigate("/"); // Redireciona para /home
  };

  const handleClickDash = () => {
    navigate("/dashboard"); // Redireciona para /home
  };

  const handleClickHealth = () => {
    navigate("/health"); // Redireciona para /home
  };

  const handleClickFaq = () => {
    navigate("/faq"); // Redireciona para /home
  };

  return (
    <div
      className="h-screen flex"
      style={{ backgroundColor: "#171717E5", fontWeight: 500 }}
    >
      {/* Sidebar */}
      <aside
        className="w-80 h-screen flex flex-col"
        style={{ backgroundColor: "#232323", color: "#828282", paddingLeft: "1rem" }}
      >
        {/* Logo */}
        <div className="flex justify-center -mb-4 ">
          <img src="/images/logo.png" alt="Logo" className="h-40 w-auto" />
        </div>

        {/* Sidebar Items */}
        <ul className="space-y-8 ml-8">
          <li
            className="flex items-center justify-between w-full cursor-pointer"
            onClick={handleClickHome}
          >
            <div className="flex items-center space-x-4">
              <FaHome className="text-gray-400" size={24} />
              <span style={{ color: "#828282", fontSize: "18px" }}>Home</span>
            </div>
            <FaChevronRight
              className="text-gray-400"
              size={16}
              style={{ marginLeft: "auto", marginRight: "25px" }}
            />
          </li>
          <li
            className="flex items-center justify-between w-full cursor-pointer"
            onClick={handleClickDash}
          >
            <div className="flex items-center space-x-4">
              <FaChartBar className="text-gray-400" size={24} />
              <span style={{ color: "#828282", fontSize: "18px" }}>Dashboard</span>
            </div>
            <FaChevronRight
              className="text-gray-400"
              size={16}
              style={{ marginLeft: "auto", marginRight: "25px" }}
            />
          </li>
          <li
            className="flex items-center justify-between w-full cursor-pointer"
            onClick={handleClickHealth}
          >
            <div className="flex items-center space-x-4">
              <FaMap className="text-gray-400" size={24} />
              <span style={{ color: "#828282", fontSize: "18px" }}>Health Map</span>
            </div>
            <FaChevronRight
              className="text-gray-400"
              size={16}
              style={{ marginLeft: "auto", marginRight: "25px" }}
            />
          </li>
          <li
            className="flex items-center justify-between w-full cursor-pointer"
            onClick={handleClickFaq}
          >
            <div className="flex items-center space-x-4">
              <FaQuestionCircle className="text-green-500" size={24} />
              <span style={{ color: "#1C6E14", fontSize: "18px" }}>Help</span>
            </div>
            <FaChevronRight
              className="text-gray-400"
              size={16}
              style={{ marginLeft: "auto", marginRight: "25px" }}
            />
          </li>
        </ul>
      </aside>

      {/* Main content with FAQ and videos */}
      <main className="w-4/5 h-screen p-10">
        <div className="text-white text-xl font-semibold mb-6">
          Frequently Asked Questions (Faq)
        </div>

        {/* FAQ Section */}
        <div className="space-y-2 mb-6">
          {[
            "How do I monitor my crops on FieldSentinel?",
            "How do I interpret the analyses?",
            "How can I conduct more accurate real-time analyses?",
            "How can I manage water resources using FieldSentinel?",
            "How do I set up the drone?",
            "How can I contact the FieldSentinel team?",
          ].map((faq, index) => (
            <div
              key={index}
              className="bg-opacity-50 text-white text-lg p-2 rounded-md cursor-pointer"
              style={{ backgroundColor: "#7BCE11A3" }} // Cor verde customizada
            >
              {faq}
            </div>
          ))}
        </div>

        {/* Videos Section */}
        <div className="grid grid-cols-2 gap-6">
          {Array(2)
            .fill()
            .map((_, index) => (
              <div key={index} className="space-y-2">
                <div className="text-white text-lg">
                  {index % 2 === 0
                    ? "Learn how to perform your analyses"
                    : "Learn how to configure FieldSentinel"}
                </div>
                <div className="relative">
                  <img
                    src="/images/thumbnail.png"
                    alt="Video thumbnail"
                    className="w-full h-auto rounded-xl" // Tamanho das imagens restaurado
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <FaPlayCircle className="text-black text-9xl" /> {/* Ícone maior e preto */}
                  </div>
                </div>
              </div>
            ))}
        </div>
      </main>
    </div>
  );
}

export default Faq;
