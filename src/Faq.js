import React, { useState } from "react";
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
    navigate('/'); // Redireciona para /home
  };

  const handleClickDash = () => {
    navigate('/dashboard'); // Redireciona para /dashboard
  };

  const handleClickHealth = () => {
    navigate('/health'); // Redireciona para /health
  };

  const handleClickFaq = () => {
      navigate('/faq'); // Redireciona para /faq
    };

  const [messages, setMessages] = useState([
    { text: "Hey! How can I help you today?", type: "bot" },
    { text: "Select one of the options to learn more about them below!", type: "bot" },
  ]);
  const [showButtons, setShowButtons] = useState(true); // Estado para controlar a visibilidade dos botões

  // Função para simular o envio de uma mensagem selecionada (opção 1)
  const handleSelectOption1 = () => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: "You selected: Learn more about the analyses", type: "user" },
      { text: "The analyses provide insight into crop health and soil conditions.", type: "bot" },
    ]);
    setShowButtons(false); // Esconde os botões após a seleção
  };

  // Função para simular o envio de uma mensagem selecionada (opção 2)
  const handleSelectOption2 = () => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: "You selected: Learn more about drones", type: "user" },
      { text: "Drones will help you monitor large areas of crops more efficiently.", type: "bot" },
    ]);
    setShowButtons(false); // Esconde os botões após a seleção
  };

  return (
    <div
      className="h-screen flex"
      style={{ backgroundColor: "#171717E5", fontWeight: 500 }}
    >
      {/* Sidebar */}
      <aside
        className="w-80 h-screen flex flex-col relative"
        style={{ backgroundColor: "#232323", color: "#828282", paddingLeft: "1rem" }}
      >
        {/* Logo */}
        <div className="flex justify-center -mb-4 ">
          <img src="/images/logo.png" alt="Logo" className="h-40 w-auto" />
        </div>

        {/* Sidebar Items */}
        <ul className="space-y-8 ml-8 mb-12">
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

        {/* Chatbot */}
        <div
          className="bg-gray-800 p-4 rounded-lg flex flex-col justify-between w-72 min-h-96"
          style={{ position: "absolute", left: "5%", right: "10px", top: '44%' }}
        >
          <div className="mb-2">
            <div className="bg-green-500 text-white text-center p-2 rounded-t-lg">
              I can help!
            </div>
            <div className="bg-gray-900 p-2 space-y-2 min-h-80 overflow-y-auto rounded-b-lg">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`p-2 rounded-lg ${
                    message.type === "bot"
                      ? "bg-green-600 text-white"
                      : "bg-gray-700 text-white"
                  }`}
                >
                  {message.text}
                </div>
              ))}
            </div>
          </div>

          {/* Botões que somem após a seleção */}
          {showButtons && (
            <div className="flex justify-around space-x-4">
              <button
                className="bg-green-500 p-2 rounded-lg text-white"
                onClick={handleSelectOption1}
              >
                Analysis
              </button>
              <button
                className="bg-green-500 p-2 rounded-lg text-white"
                onClick={handleSelectOption2}
              >
                Drones
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="w-4/5 h-screen p-10">
        <div className="text-white text-xl font-semibold mb-6">
          Frequently Asked Questions (Faq)
        </div>

        {/* FAQ Section */}
        <div className="space-y-2 mb-6">
          {[
            "How do I monitor my crops on FielSentinel?",
            "How do I interpret the analyses?",
            "How can I conduct more accurate real-time analyses?",
            "How can I manage water resources using FielSentinel?",
            "How do I set up the drone?",
            "How can I contact the FielSentinel team?",
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
                    : "Learn how to configure FielSentinel"}
                </div>
                <div className="relative">
                  <img
                    src="/images/thumbnail.png"
                    alt="Video thumbnail"
                    className="w-full h-auto"
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <FaPlayCircle className="text-black text-7xl" />
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
